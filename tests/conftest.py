import pytest
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from actor.actor_system import ActorSystem
from scripts.generate_certs import generate_self_signed_cert

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
async def actor_system():
    system = ActorSystem("test-system")
    yield system
    await system.shutdown()


@pytest.fixture
async def actor_system_with_server():
    system = ActorSystem("test-system", host="localhost", port=0)
    await system.start_server()
    yield system
    await system.shutdown()


@pytest.fixture
async def two_actor_systems():
    system1 = ActorSystem("system-1", host="localhost", port=0)
    system2 = ActorSystem("system-2", host="localhost", port=0)
    
    await system1.start_server()
    await system2.start_server()
    
    yield system1, system2
    
    await system1.shutdown()
    await system2.shutdown()


@pytest.fixture
def ssl_cert_files():
    key_file, cert_file = generate_self_signed_cert()
    
    if key_file is None or cert_file is None:
        pytest.skip("OpenSSL not available for certificate generation")
    
    return cert_file, key_file


@pytest.fixture
async def ssl_actor_systems(ssl_cert_files):
    cert_file, key_file = ssl_cert_files
    
    system1 = ActorSystem("ssl-system-1", host="localhost", port=0)
    system2 = ActorSystem("ssl-system-2", host="localhost", port=0)
    
    system1.enable_ssl(cert_file, key_file)
    system2.enable_ssl(cert_file, key_file)
    
    await system1.start_server()
    await system2.start_server()
    
    yield system1, system2
    
    await system1.shutdown()
    await system2.shutdown()
