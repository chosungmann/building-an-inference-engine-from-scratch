package book.badak;

import android.content.res.AssetFileDescriptor;

/**
 * badak의 함수들 (44장).
 *
 * <p>{@code native}라고 적은 메서드의 몸은 libbadak_ffi.so 안에 있다
 * (crates/badak_ffi/src/jni.rs). 이름이 {@code Java_book_badak_Badak_open}의
 * 꼴이어서 JVM이 찾아낸다. 패키지나 클래스의 이름을 바꾸면 그쪽의 이름도
 * 바꿔야 한다.
 */
public final class Badak {
    static {
        // lib/arm64-v8a/libbadak_ffi.so 를 찾아 올린다.
        System.loadLibrary("badak_ffi");
    }

    private Badak() {}

    public static native String version();

    public static native String cpuFeatures();

    static native String lastError();

    static native long open(int fd, long offset, long length);

    static native int classifyJpeg(long handle, byte[] jpeg, int[] indices, float[] probabilities);

    static native int run(long handle, float[] input, float[] output);

    static native void close(long handle);

    /** 받은 모델 파일을 확인하고 놓는다. 오류의 종류를 돌려준다(0이면 성공). */
    public static native int install(String directory, String manifest, String downloaded);

    /** 지금 쓰는 판의 모델 파일의 길. 놓인 것이 없으면 null. */
    public static native String current(String directory, String name);

    static native long llmOpen(String directory, int weights);

    static native int llmChat(
            long handle, String system, String question, int maxTokens, PieceListener listener);

    static native void llmClose(long handle);

    // ANCHOR: llm
    /**
     * 글 조각을 받는 쪽 (47장). 엔진이 이 메서드를 거꾸로 부른다.
     *
     * <p>조각은 온전한 UTF-8 바이트다. 글(String)이 아니라 바이트로 받는
     * 까닭은 jni.rs에 적었다. 거짓을 돌려주면 엔진이 그만 쓴다.
     */
    public interface PieceListener {
        boolean onPiece(byte[] utf8);
    }

    /** 언어 모델. 다 쓰면 {@link #close}를 불러야 한다. */
    public static final class Llm implements AutoCloseable {
        private long handle;

        /** 모델 폴더(config.json, model.safetensors, tokenizer.json)를 연다. */
        public Llm(String directory) {
            // 0: 파일에 적힌 그대로. 파일을 메모리에 비춘다.
            handle = llmOpen(directory, 0);
            if (handle == 0) {
                throw new IllegalStateException(lastError());
            }
        }

        /**
         * 질문 하나에 답한다. 쓴 토큰의 수를 돌려준다. 답을 다 쓸 때까지 돌아오지
         * 않는다. system은 대화의 맨 앞에 놓는 지시문이고 null이면 넣지 않는다.
         */
        public synchronized int chat(
                String system, String question, int maxTokens, PieceListener listener) {
            if (handle == 0) {
                throw new IllegalStateException("닫힌 모델입니다");
            }
            int written = llmChat(handle, system, question, maxTokens, listener);
            if (written < 0) {
                throw new IllegalStateException(lastError());
            }
            return written;
        }

        @Override
        public synchronized void close() {
            if (handle != 0) {
                llmClose(handle);
                handle = 0;
            }
        }
    }
    // ANCHOR_END: llm

    // ANCHOR: classifier
    /** 분류 결과 하나. */
    public static final class Score {
        public final int index;
        public final float probability;

        Score(int index, float probability) {
            this.index = index;
            this.probability = probability;
        }
    }

    /**
     * 분류기. 엔진 쪽의 분류기를 가리키는 주소를 {@code long}으로 들고 있다.
     *
     * <p>쓰레기 수집기는 엔진이 잡은 메모리를 모른다. 다 쓰면 {@link #close}를
     * 불러야 한다. try-with-resources로 쓰면 잊지 않는다.
     */
    public static final class Classifier implements AutoCloseable {
        private long handle;

        /** 앱 번들에 든 모델을 꺼내지 않고 연다. */
        public Classifier(AssetFileDescriptor asset) {
            handle = open(
                    asset.getParcelFileDescriptor().getFd(),
                    asset.getStartOffset(),
                    asset.getLength());
            if (handle == 0) {
                throw new IllegalStateException(lastError());
            }
        }

        /** JPEG 파일의 바이트를 받아 확률이 높은 부류부터 돌려준다. */
        public synchronized Score[] classify(byte[] jpeg, int count) {
            if (handle == 0) {
                throw new IllegalStateException("닫힌 분류기입니다");
            }
            int[] indices = new int[count];
            float[] probabilities = new float[count];
            int written = classifyJpeg(handle, jpeg, indices, probabilities);
            if (written < 0) {
                throw new IllegalStateException(lastError());
            }
            Score[] top = new Score[written];
            for (int rank = 0; rank < written; rank++) {
                top[rank] = new Score(indices[rank], probabilities[rank]);
            }
            return top;
        }

        @Override
        public synchronized void close() {
            if (handle != 0) {
                Badak.close(handle);
                handle = 0;
            }
        }
    }
    // ANCHOR_END: classifier
}
