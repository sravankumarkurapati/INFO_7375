#!/usr/bin/env python3
"""
Test Hugging Face authentication
"""

from huggingface_hub import login, whoami
from dotenv import load_dotenv
import os

print("="*80)
print("🔐 TESTING HUGGING FACE AUTHENTICATION")
print("="*80)

# Load environment variables from .env
load_dotenv()

# Get token
token = os.getenv('HF_TOKEN')

if not token or token == 'hf_paste_your_actual_token_here':
    print("❌ HF_TOKEN not configured in .env file")
    print("\n📝 Steps to fix:")
    print("1. Edit .env file: nano .env")
    print("2. Replace placeholder with your real token")
    print("3. Save and try again")
    exit(1)

print(f"🔑 Token found: {token[:10]}...{token[-5:]}")
print("🔄 Logging in...")

try:
    # Login
    login(token=token)
    
    # Verify
    user = whoami()
    
    print("\n✅ Login successful!")
    print(f"👤 Username: {user['name']}")
    print(f"📧 Email: {user.get('email', 'Not provided')}")
    print(f"🔗 Profile: https://huggingface.co/{user['name']}")
    
    print("\n" + "="*80)
    print("🎉 Hugging Face authentication working!")
    
except Exception as e:
    print(f"\n❌ Login failed: {e}")
    print("\n📝 Steps to fix:")
    print("1. Check your token at: https://huggingface.co/settings/tokens")
    print("2. Make sure token has 'Write' permission")
    print("3. Update .env file with correct token")
    exit(1)
