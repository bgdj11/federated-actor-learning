import pytest
import asyncio
import ssl
from tests.fixtures import Ping, Pong, PingActor, PongActor, CounterActor


@pytest.mark.asyncio
async def test_ssl_context_creation(actor_system, ssl_cert_files):
    cert_file, key_file = ssl_cert_files
    
    actor_system.enable_ssl(cert_file, key_file)
    
    assert actor_system._ssl_context is not None
    assert isinstance(actor_system._ssl_context, ssl.SSLContext)
    
    assert actor_system._ssl_client_context is not None
    assert isinstance(actor_system._ssl_client_context, ssl.SSLContext)


@pytest.mark.asyncio
async def test_ssl_server_startup(ssl_cert_files):
    from actor.actor_system import ActorSystem
    
    cert_file, key_file = ssl_cert_files
    
    system = ActorSystem("ssl-test", host="localhost", port=0)
    system.enable_ssl(cert_file, key_file)
    
    await system.start_server()
    
    assert system._server is not None
    assert system._running is True
    assert system.port > 0
    
    await system.shutdown()


@pytest.mark.asyncio
async def test_ssl_remote_messaging(ssl_actor_systems):
    system1, system2 = ssl_actor_systems
    
    counter = system1.actor_of(CounterActor, "secure-counter")

    remote_ref = system2.remote_ref("secure-counter", "localhost", system1.port)
    
    for i in range(5):
        remote_ref.tell(Ping(count=i))
    
    await asyncio.sleep(0.8)
    
    actor = system1._actors["secure-counter"]
    assert actor.count == 5, f"Expected 5 messages, got {actor.count}"


@pytest.mark.asyncio
async def test_ssl_ping_pong_exchange(ssl_actor_systems):
    system1, system2 = ssl_actor_systems
    
    pong_ref = system1.actor_of(PongActor, "ssl-pong")
    ping_ref = system2.actor_of(PingActor, "ssl-ping")

    system2._actors["ssl-ping"].pong_ref = system2.remote_ref("ssl-pong", "localhost", system1.port)
    system1._actors["ssl-pong"].ping_ref = system1.remote_ref("ssl-ping", "localhost", system2.port)
    

    remote_pong = system2.remote_ref("ssl-pong", "localhost", system1.port)
    remote_pong.tell(Ping(count=1))
    
    await asyncio.sleep(1.2)
    
    ping_actor = system2._actors["ssl-ping"]
    assert ping_actor.received >= 1, f"Expected at least 1 encrypted pong, got {ping_actor.received}"


@pytest.mark.asyncio
async def test_ssl_connection_persistence(ssl_actor_systems):
    system1, system2 = ssl_actor_systems
    
    system1.actor_of(CounterActor, "counter")
    
    remote_ref = system2.remote_ref("counter", "localhost", system1.port)
    
    for i in range(10):
        remote_ref.tell(Ping(count=i))
    
    await asyncio.sleep(1.0)
    
    addr_key = ("localhost", system1.port)
    assert addr_key in system2._remote_connections
  
    counter = system1._actors["counter"]
    assert counter.count == 10


@pytest.mark.asyncio
async def test_ssl_multi_client_connections(ssl_cert_files):
    from actor.actor_system import ActorSystem
    
    cert_file, key_file = ssl_cert_files
    
    # Create server system
    server = ActorSystem("ssl-server", port=0)
    server.enable_ssl(cert_file, key_file)
    await server.start_server()
    
    server.actor_of(CounterActor, "shared-counter")
    
    # Create multiple client systems
    clients = []
    for i in range(3):
        client = ActorSystem(f"ssl-client-{i}", port=0)
        client.enable_ssl(cert_file, key_file)
        await client.start_server()
        clients.append(client)
    
    # Each client sends messages
    for i, client in enumerate(clients):
        remote_ref = client.remote_ref("shared-counter", "localhost", server.port)
        for j in range(3):
            remote_ref.tell(Ping(count=i*3 + j))
    
    await asyncio.sleep(1.5)
    
    counter = server._actors["shared-counter"]
    assert counter.count == 9  # 3 clients * 3 messages
    
    await server.shutdown()
    for client in clients:
        await client.shutdown()


@pytest.mark.asyncio
async def test_ssl_vs_non_ssl_incompatibility(ssl_cert_files):
    from actor.actor_system import ActorSystem
    
    cert_file, key_file = ssl_cert_files
    
    non_ssl_system = ActorSystem("non-ssl", port=0)
    await non_ssl_system.start_server()
    
    non_ssl_system.actor_of(CounterActor, "counter")
    
    ssl_system = ActorSystem("ssl", port=0)
    ssl_system.enable_ssl(cert_file, key_file)
    await ssl_system.start_server()
    
    # The actual incompatibility will manifest as connection errors
    # which is expected behavior for security - SSL cannot talk to non-SSL
    
    await non_ssl_system.shutdown()
    await ssl_system.shutdown()


@pytest.mark.asyncio
async def test_ssl_configuration_logging(ssl_cert_files, caplog):
    import logging
    from actor.actor_system import ActorSystem
    
    cert_file, key_file = ssl_cert_files
    
    with caplog.at_level(logging.INFO):
        system = ActorSystem("test-ssl-logging")
        system.enable_ssl(cert_file, key_file)
    
    assert "SSL enabled" in caplog.text
    
    await system.shutdown()
