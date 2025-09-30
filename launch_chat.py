#!/usr/bin/env python3
"""
Quick launcher for ARM Chat application.

Usage:
    python launch_chat.py
"""

from arm_chat import create_chat_interface

if __name__ == "__main__":
    print("🗨️  Starting ARM Chat...")
    print("📂 Load a saved manifold file to begin chatting")
    print("🌐 Interface will open at: http://127.0.0.1:7861")
    print()
    
    interface = create_chat_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )
