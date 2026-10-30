"""Loopback-only production build preview with fixed local model files and live CSP."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        clean=urlsplit(path).path
        if clean in ('/benchmark','/benchmark/'):return str(Path(self.directory)/'index.html')
        if clean.startswith('/models/portable-v1/'):
            name=clean.removeprefix('/models/portable-v1/')
            if name in ('manifest.json','detector.onnx','classifier.onnx'):return str(self.server.models/name)
            return str(Path(self.directory)/'missing-model')
        return super().translate_path(path)
    def end_headers(self):
        clean=urlsplit(self.path).path
        if clean in ('/benchmark','/benchmark/') or (clean.startswith('/assets/local.worker-') and clean.endswith('.js')):
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
            self.send_header('Cross-Origin-Opener-Policy','same-origin')
            self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cross-Origin-Resource-Policy','same-origin');super().end_headers()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--port',type=int,default=4174);a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(Path('dist').resolve())))
    server.models=a.models.resolve();server.serve_forever()
