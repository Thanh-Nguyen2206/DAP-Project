"""
AuthManager: Simple authentication system for the stock dashboard
"""

import streamlit as st
from typing import Dict


class AuthManager:
    """Simple authentication manager for demo purposes."""

    DEMO_USERS: Dict[str, Dict[str, str]] = {
        "demo": {"password": "demo123", "role": "user"}
    }

    def __init__(self):
        if 'auth_initialized' not in st.session_state:
            st.session_state.auth_initialized = True
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None

    @staticmethod
    def init_auth() -> 'AuthManager':
        """Factory helper to create and initialize AuthManager."""
        return AuthManager()

    def logout(self) -> None:
        """Log out the current user and refresh the app."""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.rerun()  # Updated from st.experimental_rerun()

    def login_form(self) -> bool:
        """Render login form in the sidebar and handle authentication.

        Returns True when the user is authenticated (or already was).
        """
        if st.session_state.get('authenticated'):
            st.sidebar.success(
                f"Logged in as {st.session_state.get('username')} ({st.session_state.get('user_role')})"
            )
            if st.sidebar.button("Logout", key="auth_logout_btn"):
                self.logout()
            return True

        st.sidebar.markdown("###  Login")
        st.sidebar.info("Demo account:\n- demo / demo123")

        username = st.sidebar.text_input("Username", key="auth_username_input", value="")
        password = st.sidebar.text_input("Password", type="password", key="auth_password_input", value="")

        if st.sidebar.button("Login", key="auth_login_btn"):
            # Debug output
            st.sidebar.write(f"Debug - Username: '{username}', Password length: {len(password) if password else 0}")
            
            if self._verify_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = self._get_user_role(username)
                st.sidebar.success("Login successful")
                st.rerun()  # Updated from st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials - Please check username and password")
        return False

    def _verify_credentials(self, username: str, password: str) -> bool:
        # Debug logging
        print(f"Debug - Checking credentials for username: '{username}', password: '{password}'")
        print(f"Available users: {list(self.DEMO_USERS.keys())}")
        
        if not username or not password:
            print("Debug - Username or password is empty")
            return False
        
        # Strip whitespace
        username = username.strip()
        password = password.strip()
        
        user = self.DEMO_USERS.get(username)
        if not user:
            print(f"Debug - User '{username}' not found")
            return False
        
        result = password == user.get('password')
        print(f"Debug - Password match result: {result}")
        return result

    def _get_user_role(self, username: str) -> str:
        user = self.DEMO_USERS.get(username, {})
        return user.get('role', 'user')

    def require_auth(self) -> bool:
        if not st.session_state.get('authenticated'):
            st.error("Please log in to access this page.")
            return False
        return True

    def is_admin(self) -> bool:
        return st.session_state.get('authenticated', False) and st.session_state.get('user_role') == 'admin'