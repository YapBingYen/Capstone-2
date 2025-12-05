#!/usr/bin/env python3
"""
Pet ID Malaysia - Application Launcher
Simple startup script for the Pet Recognition Web Application

Usage: python run.py [options]

Options:
    --debug     Run in debug mode
    --port N    Specify port number (default: 5000)
    --host H    Specify host address (default: 0.0.0.0)
"""

import sys
import os
import argparse
from app import app, initialize_system

def main():
    """Main function to parse arguments and run the application"""
    parser = argparse.ArgumentParser(description='Pet ID Malaysia - AI Pet Recognition System')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=8080, help='Port number (default: 8080)')
    parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')

    args = parser.parse_args()

    print("🚀 Starting Pet ID Malaysia - AI Pet Recognition System")
    print("=" * 60)

    # Initialize the system
    if not initialize_system():
        print("❌ Failed to initialize system. Exiting...")
        sys.exit(1)

    print("✅ System initialized successfully!")
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print("📱 Open your browser and navigate to the address above")
    print("⚠️  Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Run the Flask application
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port,
            use_reloader=False if not args.debug else True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()