#!/usr/bin/env python3

import sqlite3
import uuid
from werkzeug.security import generate_password_hash

def create_test_user():
    print("🔧 Creating test user for login...")
    
    try:
        conn = sqlite3.connect('/Users/rishi/Documents/plant1/backend/users.db')
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("Creating users table...")
            cursor.execute('''
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Create test user
        test_email = "test@example.com"
        test_password = "password123"
        test_name = "Test User"
        
        # Check if test user already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (test_email,))
        if cursor.fetchone():
            print(f"✅ Test user {test_email} already exists")
        else:
            # Create new test user
            user_id = str(uuid.uuid4())
            password_hash = generate_password_hash(test_password)
            
            cursor.execute('''
                INSERT INTO users (id, name, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (user_id, test_name, test_email, password_hash))
            
            print(f"✅ Created test user:")
            print(f"   Email: {test_email}")
            print(f"   Password: {test_password}")
            print(f"   Name: {test_name}")
        
        # List all users
        cursor.execute('SELECT name, email FROM users')
        users = cursor.fetchall()
        print(f"\n📋 All users in database:")
        for user in users:
            print(f"   {user[1]} ({user[0]})")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎯 You can now login with:")
        print(f"   Email: {test_email}")
        print(f"   Password: {test_password}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_test_user()