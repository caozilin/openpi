"""Bounded transport recovery; never advance physics or change an inference packet."""
import time

class ResilientConnection:
    def __init__(self, factory, args, kwargs, *, retry_errors, max_retries=3):
        self.factory=factory;self.args=args;self.kwargs=kwargs;self.retry_errors=retry_errors
        self.max_retries=max_retries;self.conn=None;self.metadata=None;self.packet=None
        self.deadline=None;self.retries=0
    def __enter__(self):
        self.conn=self.factory(*self.args,**self.kwargs).__enter__();return self
    def __exit__(self,*args):
        if self.conn is not None:return self.conn.__exit__(*args)
    def _remaining(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Original600s inference response deadline expired')
        return remaining
    def _recover(self):
        if self.retries>=self.max_retries:raise RuntimeError('Bounded WebSocket reconnect attempts exhausted')
        self.retries+=1;remaining=self._remaining()
        try:self.conn.close()
        except Exception:pass
        time.sleep(min(float(self.retries),remaining))
        options=dict(self.kwargs);options['open_timeout']=min(float(options.get('open_timeout',300)),self._remaining())
        self.conn=self.factory(*self.args,**options).__enter__()
        metadata=self.conn.recv(timeout=self._remaining())
        if metadata!=self.metadata:raise RuntimeError('Server identity changed on reconnect; refusing inference replay')
        print(f'TRANSPORT_RECONNECT retry={self.retries}; identical metadata and packet; remaining={self._remaining():.1f}s',flush=True)
    def send(self, packet):
        self.packet=packet;self.deadline=time.monotonic()+600.;self.retries=0
        while True:
            try:self.conn.send(packet);return
            except TimeoutError:raise
            except self.retry_errors:
                while True:
                    try:self._recover();break
                    except TimeoutError:raise
                    except self.retry_errors:continue
    def recv(self, timeout=600):
        if self.metadata is None:
            self.metadata=self.conn.recv(timeout=timeout);return self.metadata
        while True:
            try:return self.conn.recv(timeout=min(float(timeout),self._remaining()))
            except TimeoutError:raise
            except self.retry_errors:
                while True:
                    try:self._recover();self.conn.send(self.packet);break
                    except TimeoutError:raise
                    except self.retry_errors:continue


def install():
    import websockets.sync.client
    import websockets.exceptions
    original=websockets.sync.client.connect
    def connect(*args,**kwargs):
        return ResilientConnection(original,args,kwargs,retry_errors=(websockets.exceptions.ConnectionClosed,OSError))
    websockets.sync.client.connect=connect
