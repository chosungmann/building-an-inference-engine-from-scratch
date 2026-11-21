//! 여러 요청을 한 번의 순전파에 함께 태우는 스케줄러 (40장).
//!
//! 39장의 엔진은 자물쇠 하나로 요청을 줄 세웠다. 여기서는 스레드 하나가
//! 모델을 쥐고 **스텝**을 되풀이한다. 한 스텝은 순전파 한 번이고, 그
//! 순전파에 지금 돌고 있는 모든 요청의 새 토큰이 함께 들어간다.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use badak_formats::{Error, Result};
use badak_models::constrain::Constraint;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::paged::{PAGE_TOKENS, PagePool, PagedCache, Step};
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::chat::Message;
use badak_models::tokenizer::{StreamDecoder, TokenId, Tokenizer};

use crate::engine::{Completion, Engine, Finish, Options, common_prefix};

// ANCHOR: policy
/// 요청을 어떻게 묶을지.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// 정적 배칭. 요청이 `size`개 모이거나 첫 요청이 `wait`만큼 기다리면
    /// 그때까지 모인 것을 함께 시작한다. **그 배치가 다 끝날 때까지 새
    /// 요청을 받지 않는다.**
    Static {
        /// 배치의 크기.
        size: usize,
        /// 배치가 차기를 기다리는 한도.
        wait: Duration,
    },
    /// 연속 배칭. 스텝마다 자리가 비어 있으면 기다리던 요청을 바로 태운다.
    /// 끝난 요청은 그 스텝에 내린다.
    Continuous {
        /// 함께 도는 요청의 한도.
        max: usize,
    },
}

/// 스케줄러의 설정.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// 요청을 어떻게 묶을지.
    pub policy: Policy,
    /// 글을 읽는 요청이 한 스텝에 넣는 토큰의 한도. 긴 글 하나가 한 스텝을
    /// 길게 만들어 다른 요청들의 토큰이 늦어지는 것을 막는다.
    pub chunk: usize,
    /// KV 캐시의 페이지의 한도.
    pub pages: usize,
    /// 끝난 요청의 캐시를 몇 개까지 남겨 둘지. 다음 요청이 앞부분을 빌린다.
    pub keep: usize,
}
// ANCHOR_END: policy

/// 스케줄러가 요청 하나에 돌려보내는 것.
enum Event {
    Text(String),
    Done(Box<Completion>),
    Failed(String),
}

/// 기다리는 요청 하나.
struct Job {
    ids: Vec<TokenId>,
    options: Options,
    events: Sender<Event>,
    /// 받는 쪽이 연결을 끊었으면 참이 된다.
    cancelled: Arc<AtomicBool>,
    arrived: Instant,
    tokenize: f64,
}

/// 돌고 있는 요청 하나.
struct Running {
    job: Job,
    cache: PagedCache,
    /// 캐시에 넣은 토큰들.
    fed: Vec<TokenId>,
    /// 아직 넣지 않은 토큰들. 글을 읽는 동안은 프롬프트의 나머지이고, 답을
    /// 쓰는 동안은 방금 고른 토큰 하나다.
    pending: Vec<TokenId>,
    /// 프롬프트와 지금까지 쓴 토큰. 샘플러가 본다.
    history: Vec<TokenId>,
    sampler: Sampler,
    /// 답이 문법에 묶였으면 그 상태 (41장).
    constraint: Option<Constraint>,
    decoder: StreamDecoder,
    text: String,
    written: usize,
    cached: usize,
    queued: f64,
    started: Instant,
    first_token: Option<f64>,
    finish: Option<Finish>,
}

/// 끝난 요청이 남긴 캐시.
struct Idle {
    tokens: Vec<TokenId>,
    cache: PagedCache,
    /// 마지막으로 쓰인 스텝.
    used: u64,
}

/// 밖에서 들여다볼 수 있는 숫자들.
#[derive(Debug, Default)]
pub struct Stats {
    /// 지금 돌고 있는 요청의 수.
    pub running: AtomicUsize,
    /// 남겨 둔 캐시의 수.
    pub idle: AtomicUsize,
    /// 순전파를 돈 횟수.
    pub steps: AtomicUsize,
    /// 그 순전파들에 넣은 줄(토큰)의 합.
    pub rows: AtomicUsize,
}

/// 요청을 받아 스케줄러의 스레드로 넘기는 쪽.
pub struct Scheduler {
    /// 모델의 이름.
    pub name: String,
    /// 읽는 글과 쓰는 글을 더한 길이의 한도(토큰).
    pub context: usize,
    /// 설정.
    pub settings: Settings,
    /// 숫자들.
    pub stats: Arc<Stats>,
    /// 페이지의 풀.
    pub pool: PagePool,
    jobs: Mutex<Sender<Job>>,
    engine: Arc<Engine>,
}

impl Scheduler {
    /// 엔진의 모델로 스케줄러를 띄운다. 스레드 하나가 모델을 쥐고 돈다.
    pub fn start(engine: Arc<Engine>, settings: Settings) -> Self {
        let (model, tokenizer) = engine.parts();
        let pool = PagePool::new(&model.config, settings.pages);
        let stats = Arc::new(Stats::default());
        let (jobs, queue) = mpsc::channel();
        let mut state = State {
            engine: Arc::clone(&engine),
            model,
            tokenizer,
            settings,
            pool: pool.clone(),
            stats: Arc::clone(&stats),
            running: Vec::new(),
            idle: Vec::new(),
            step: 0,
        };
        std::thread::spawn(move || state.run(&queue));
        Self {
            name: engine.name.clone(),
            context: engine.context,
            settings,
            stats,
            pool,
            jobs: Mutex::new(jobs),
            engine,
        }
    }

    /// 대화에 답을 쓴다. 39장의 `Engine::complete`와 쓰는 법이 같다.
    pub fn complete(
        &self,
        messages: &[Message],
        options: &Options,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion> {
        let arrived = Instant::now();
        let ids = self.engine.encode(messages);
        let tokenize = arrived.elapsed().as_secs_f64();
        if ids.len() + options.max_tokens > self.context {
            return Err(Error::Unsupported(format!(
                "읽을 글({}토큰)과 쓸 글({}토큰)을 더하면 한도({}토큰)를 넘습니다",
                ids.len(),
                options.max_tokens,
                self.context
            )));
        }
        let (events, results) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            ids,
            options: options.clone(),
            events,
            cancelled: Arc::clone(&cancelled),
            arrived,
            tokenize,
        };
        let gone = || Error::Unsupported("스케줄러가 멈췄습니다".into());
        self.jobs
            .lock()
            .map_err(|_| gone())?
            .send(job)
            .map_err(|_| gone())?;
        loop {
            match results.recv().map_err(|_| gone())? {
                Event::Text(text) => {
                    if !on_text(&text) {
                        cancelled.store(true, Ordering::Release);
                    }
                }
                Event::Done(completion) => return Ok(*completion),
                Event::Failed(message) => return Err(Error::Unsupported(message)),
            }
        }
    }
}

/// 스케줄러의 스레드가 쥐는 것.
struct State {
    engine: Arc<Engine>,
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    settings: Settings,
    pool: PagePool,
    stats: Arc<Stats>,
    running: Vec<Running>,
    idle: Vec<Idle>,
    step: u64,
}

impl State {
    // ANCHOR: run
    /// 요청이 있는 동안 스텝을 되풀이한다.
    fn run(&mut self, queue: &Receiver<Job>) {
        loop {
            // 1. 태운다. 무엇을 언제 태울지가 두 방법의 차이다.
            if !self.admit(queue) {
                return;
            }
            // 2. 한 스텝. 돌고 있는 모든 요청이 순전파 한 번을 함께 지난다.
            if let Err(error) = self.step() {
                for mut running in self.running.drain(..) {
                    let _ = running.job.events.send(Event::Failed(error.to_string()));
                    running.finish = Some(Finish::Cancelled);
                }
            }
            // 3. 끝난 요청을 내린다. 캐시는 다음 요청이 빌려 쓰게 남겨 둔다.
            self.retire();
            self.stats
                .running
                .store(self.running.len(), Ordering::Release);
            self.stats.idle.store(self.idle.len(), Ordering::Release);
        }
    }
    // ANCHOR_END: run

    // ANCHOR: admit
    /// 기다리는 요청을 태운다. 보내는 쪽이 모두 사라졌으면 거짓이다.
    fn admit(&mut self, queue: &Receiver<Job>) -> bool {
        match self.settings.policy {
            Policy::Continuous { max } => {
                // 자리가 있는 만큼, 와 있는 요청을 바로 태운다.
                while self.running.len() < max {
                    match queue.try_recv() {
                        Ok(job) => self.start(job),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => return !self.running.is_empty(),
                    }
                }
                // 아무도 돌고 있지 않으면 다음 요청이 올 때까지 잔다.
                if self.running.is_empty() {
                    match queue.recv() {
                        Ok(job) => self.start(job),
                        Err(_) => return false,
                    }
                }
            }
            Policy::Static { size, wait } => {
                // 배치가 다 끝난 뒤에만 새 배치를 만든다.
                if !self.running.is_empty() {
                    return true;
                }
                let Ok(first) = queue.recv() else {
                    return false;
                };
                let deadline = first.arrived + wait;
                self.start(first);
                while self.running.len() < size {
                    let left = deadline.saturating_duration_since(Instant::now());
                    match queue.recv_timeout(left) {
                        Ok(job) => self.start(job),
                        Err(RecvTimeoutError::Timeout) => break,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            }
        }
        true
    }
    // ANCHOR_END: admit

    // ANCHOR: start
    /// 요청 하나를 태운다. 남겨 둔 캐시 가운데 앞부분이 가장 길게 겹치는
    /// 것에서 그 앞부분을 빌린다. 베끼지 않고 페이지를 함께 가리킨다.
    fn start(&mut self, job: Job) {
        let best = self
            .idle
            .iter_mut()
            .map(|idle| (common_prefix(&idle.tokens, &job.ids), idle))
            .max_by_key(|(common, _)| *common);
        // 39장과 같이 마지막 토큰 하나는 늘 다시 계산한다.
        let limit = job.ids.len().saturating_sub(1);
        let (cache, cached) = match best {
            Some((common, idle)) if job.options.reuse && common > 0 => {
                idle.used = self.step;
                let cached = common.min(limit);
                (idle.cache.fork(cached), cached)
            }
            _ => (PagedCache::new(&self.pool), 0),
        };
        let sampler = Sampler::new(SamplerConfig {
            temperature: job.options.temperature,
            top_p: job.options.top_p,
            seed: job.options.seed,
            ..SamplerConfig::default()
        });
        self.running.push(Running {
            queued: job.arrived.elapsed().as_secs_f64() - job.tokenize,
            fed: job.ids[..cached].to_vec(),
            pending: job.ids[cached..].to_vec(),
            history: job.ids.clone(),
            cache,
            cached,
            sampler,
            constraint: self.engine.constraint(&job.options),
            decoder: StreamDecoder::default(),
            text: String::new(),
            written: 0,
            started: Instant::now(),
            first_token: None,
            finish: None,
            job,
        });
    }
    // ANCHOR_END: start

    /// 이번 스텝에 새로 받아야 하는 페이지를 댈 수 있게, 남겨 둔 캐시를
    /// 오래된 것부터 버린다.
    fn make_room(&mut self, takes: &[usize]) {
        loop {
            let needed: usize = self
                .running
                .iter()
                .zip(takes)
                .map(|(running, take)| running.cache.pages_needed(*take))
                .sum();
            if needed <= self.pool.available() || self.idle.is_empty() {
                return;
            }
            let oldest = (0..self.idle.len())
                .min_by_key(|&index| self.idle[index].used)
                .unwrap_or(0);
            self.idle.swap_remove(oldest);
        }
    }

    // ANCHOR: step
    /// 한 스텝. 돌고 있는 요청마다 새 토큰을 내고, 순전파 한 번을 돌고,
    /// 답을 쓰는 요청마다 다음 토큰을 고른다.
    fn step(&mut self) -> Result<()> {
        if self.running.is_empty() {
            return Ok(());
        }
        self.step += 1;
        // 글을 읽는 요청은 `chunk`개까지, 답을 쓰는 요청은 하나를 낸다.
        let takes: Vec<usize> = self
            .running
            .iter()
            .map(|running| running.pending.len().min(self.settings.chunk))
            .collect();
        self.make_room(&takes);

        let mut steps: Vec<Step<'_>> = self
            .running
            .iter_mut()
            .zip(&takes)
            .map(|(running, take)| Step {
                ids: &running.pending[..*take],
                cache: &mut running.cache,
            })
            .collect();
        let logits = self.model.forward_batch(&mut steps)?;
        drop(steps);
        self.stats.steps.fetch_add(1, Ordering::Relaxed);
        self.stats
            .rows
            .fetch_add(takes.iter().sum(), Ordering::Relaxed);

        for ((running, take), logits) in self.running.iter_mut().zip(takes).zip(&logits) {
            running.fed.extend(running.pending.drain(..take));
            if running.job.cancelled.load(Ordering::Acquire) {
                running.finish = Some(Finish::Cancelled);
                continue;
            }
            // 읽을 글이 남았으면 이번 점수는 쓰지 않는다.
            if !running.pending.is_empty() {
                continue;
            }
            // 41장: 문법에 묶였으면 지금 올 수 없는 토큰의 점수를 지운다.
            let end = &self.model.config.eos_token_ids;
            let mut logits = logits.as_slice::<f32>()?.to_vec();
            if let Some(constraint) = &mut running.constraint {
                constraint.mask(&mut logits, end);
            }
            let next = running.sampler.sample(&logits, &running.history);
            running
                .first_token
                .get_or_insert_with(|| running.started.elapsed().as_secs_f64());
            if end.contains(&next) {
                running.finish = Some(Finish::Stop);
                continue;
            }
            if let Some(constraint) = &mut running.constraint {
                constraint.accept(next);
            }
            running.history.push(next);
            running.written += 1;
            let piece = running.decoder.push(&self.tokenizer, next);
            running.text.push_str(&piece);
            if !piece.is_empty() && running.job.events.send(Event::Text(piece)).is_err() {
                running.finish = Some(Finish::Cancelled);
                continue;
            }
            if running
                .constraint
                .as_ref()
                .is_some_and(Constraint::finished)
            {
                running.finish = Some(Finish::Stop);
                continue;
            }
            if running.written >= running.job.options.max_tokens {
                running.finish = Some(Finish::Length);
                continue;
            }
            running.pending.push(next);
        }
        Ok(())
    }
    // ANCHOR_END: step

    /// 끝난 요청을 내리고, 그 캐시를 남겨 둔다.
    fn retire(&mut self) {
        let mut index = 0;
        while index < self.running.len() {
            let Some(finish) = self.running[index].finish else {
                index += 1;
                continue;
            };
            let mut done = self.running.swap_remove(index);
            let rest = done.decoder.finish();
            if !rest.is_empty() {
                done.text.push_str(&rest);
                let _ = done.job.events.send(Event::Text(rest));
            }
            let total = done.started.elapsed().as_secs_f64();
            let _ = done.job.events.send(Event::Done(Box::new(Completion {
                text: done.text,
                prompt_tokens: done.job.ids.len(),
                cached_tokens: done.cached,
                completion_tokens: done.written,
                finish,
                tokenize: done.job.tokenize,
                queued: done.queued.max(0.0),
                first_token: done.first_token.unwrap_or(total),
                total,
            })));
            if self.settings.keep == 0 {
                continue;
            }
            // 새 캐시의 앞부분일 뿐인 옛 캐시는 필요 없다. 페이지를 함께
            // 가리키고 있으므로 버려도 잃는 것이 없다.
            self.idle
                .retain(|idle| common_prefix(&idle.tokens, &done.fed) < idle.tokens.len());
            self.idle.push(Idle {
                tokens: done.fed,
                cache: done.cache,
                used: self.step,
            });
            while self.idle.len() > self.settings.keep {
                let oldest = (0..self.idle.len())
                    .min_by_key(|&index| self.idle[index].used)
                    .unwrap_or(0);
                self.idle.swap_remove(oldest);
            }
        }
    }
}

/// 한도가 `context`토큰인 요청 `slots`개가 쓸 페이지의 수.
pub fn pages_for(context: usize, slots: usize) -> usize {
    context.div_ceil(PAGE_TOKENS) * slots
}
