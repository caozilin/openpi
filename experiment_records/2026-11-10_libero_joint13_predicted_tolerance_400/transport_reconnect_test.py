from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_reconnect import ResilientConnection
import pytest
class Fake:
 def __init__(self,messages,fail_send=False):self.messages=list(messages);self.sent=[];self.fail_send=fail_send
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def close(self):pass
 def send(self,p):
  self.sent.append(p)
  if self.fail_send:self.fail_send=False;raise OSError('link closed')
 def recv(self,**kwargs):
  x=self.messages.pop(0)
  if isinstance(x,Exception):raise x
  return x

def connection(monkeypatch,first,second):
 monkeypatch.setattr('transport_reconnect.time.sleep',lambda _:None)
 queue=iter([first,second]);return ResilientConnection(lambda *a,**k:next(queue),(),{},retry_errors=(OSError,)).__enter__()
def test_recv_replay_identical_packet_and_identity(monkeypatch):
 a=Fake([b'meta',OSError('ping timeout')]);b=Fake([b'meta',b'reply'])
 c=connection(monkeypatch,a,b);assert c.recv()==b'meta';c.send(b'unchanged image/action-noise packet');assert c.recv()==b'reply'
 assert a.sent==b.sent==[b'unchanged image/action-noise packet']
def test_identity_mismatch_rejected(monkeypatch):
 c=connection(monkeypatch,Fake([b'meta',OSError()]),Fake([b'other model']))
 c.recv();c.send(b'packet')
 with pytest.raises(RuntimeError,match='identity changed'):c.recv()
def test_send_replay(monkeypatch):
 a=Fake([b'meta'],fail_send=True);b=Fake([b'meta',b'reply']);c=connection(monkeypatch,a,b)
 c.recv();c.send(b'packet');assert c.recv()==b'reply' and b.sent==[b'packet']
def test_original_deadline_preserved(monkeypatch):
 c=connection(monkeypatch,Fake([b'meta']),Fake([]));c.recv();c.send(b'packet');c.deadline=-1
 with pytest.raises(TimeoutError):c.recv()

def test_failed_reconnect_is_bounded_and_can_recover(monkeypatch):
 monkeypatch.setattr('transport_reconnect.time.sleep',lambda _:None)
 a=Fake([b'meta',OSError('ping timeout')]);b=Fake([b'meta',b'reply']);calls=iter([a,OSError('temporary connect failure'),b])
 def factory(*args,**kwargs):
  x=next(calls)
  if isinstance(x,Exception):raise x
  return x
 c=ResilientConnection(factory,(),{},retry_errors=(OSError,)).__enter__();c.recv();c.send(b'packet');assert c.recv()==b'reply' and c.retries==2

def test_retry_limit_rejected(monkeypatch):
 monkeypatch.setattr('transport_reconnect.time.sleep',lambda _:None)
 a=Fake([OSError()]);c=ResilientConnection(lambda *a,**k:a,(),{},retry_errors=(OSError,),max_retries=0)
 c.conn=a;c.metadata=b'meta';c.packet=b'p';c.deadline=10**20
 with pytest.raises(RuntimeError,match='attempts exhausted'):c.recv()
