import subprocess
import os

CERTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'certs')

def generate_self_signed_cert():
    os.makedirs(CERTS_DIR, exist_ok=True)
    
    key_file = os.path.join(CERTS_DIR, 'server.key')
    cert_file = os.path.join(CERTS_DIR, 'server.crt')
    
    if os.path.exists(key_file) and os.path.exists(cert_file):
        print(f"Certificates already exist in {CERTS_DIR}")
        return key_file, cert_file
    
    cmd = [
        'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
        '-keyout', key_file,
        '-out', cert_file,
        '-days', '365',
        '-nodes',
        '-subj', '/CN=localhost'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Generated certificates in {CERTS_DIR}")
        return key_file, cert_file
    except FileNotFoundError:
        print("OpenSSL not found. Install it or generate certificates manually.")
        print("On Windows: choco install openssl")
        print("On Ubuntu: sudo apt install openssl")
        return None, None

if __name__ == "__main__":
    generate_self_signed_cert()
