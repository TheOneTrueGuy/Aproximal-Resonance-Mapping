#!/usr/bin/env python3
"""
Launch script for the ARM Gradio interface.

Usage:
    python launch_interface.py

This will start a local web server with the ARM analysis interface.
"""

import sys
import os

def main():
    """Launch the ARM interface."""
    print("🚀 Launching ARM Interface...")
    print("=" * 50)

    # Check if we're in the virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider using: arm_env\\Scripts\\Activate.ps1")
        print()

    # Import and launch the interface
    try:
        from arm_interface import create_gradio_interface

        print("🌐 Starting web interface...")
        print("📱 Access at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop")
        print()

        interface = create_gradio_interface()
        interface.launch(
            server_name="127.0.0.1",  # Localhost only for security
            server_port=7860,
            share=False,  # Don't create public link
            show_error=True,
            quiet=False
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements-test.txt")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 ARM Interface stopped")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
