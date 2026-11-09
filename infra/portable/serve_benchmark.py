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
        for profile,directory in (('portable-v1',self.server.models),('compact-v10',self.server.compact_models)):
            prefix='/models/'+profile+'/'
            if clean.startswith(prefix):
                name=clean.removeprefix(prefix)
                if directory is not None and name in ('manifest.json','detector.onnx','classifier.onnx'):
                    return str(directory/name)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--compact-models',type=Path);p.add_argument('--port',type=int,default=4174);a=p.parse_args()
    for directory in (a.models,a.compact_models):
        if directory is not None and not all((directory/name).is_file() for name in ('manifest.json','detector.onnx','classifier.onnx')):
            p.error('Each configured profile needs all three local files')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(Path('dist').resolve())))
    server.models=a.models.resolve();server.compact_models=a.compact_models.resolve() if a.compact_models else None;server.serve_forever()
