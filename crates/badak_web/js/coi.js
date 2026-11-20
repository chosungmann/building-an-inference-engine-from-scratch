// 페이지를 "교차 출처 격리"로 만드는 서비스 워커 (48장).
//
// 스레드를 쓰려면 공유 메모리가 필요하고, 브라우저는 공유 메모리를 격리된
// 페이지에만 준다. 격리는 서버가 응답에 헤더 둘을 붙여서 알린다. GitHub
// Pages는 헤더를 고르게 해 주지 않는다.
//
// 서비스 워커는 페이지와 서버 사이에 끼어 응답을 고칠 수 있다. 여기에서 그
// 헤더들을 붙인다. 처음 한 번은 이 워커가 아직 없으므로, 페이지가 워커를
// 등록하고 스스로를 다시 읽는다.

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const request = event.request;
  // 개발자 도구가 보내는 요청 가운데 건드리면 안 되는 것이 있다.
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  event.respondWith(
    fetch(request).then((response) => {
      // 속을 볼 수 없는 응답은 고칠 수 없다.
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
