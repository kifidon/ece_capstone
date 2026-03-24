'use client';

import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react';
import { login as apiLogin, register as apiRegister, type User, type LoginResponse } from './api';
import { AUTH_TOKEN_KEY, AUTH_USER_KEY } from './auth-storage';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
    const storedUser = localStorage.getItem(AUTH_USER_KEY);
    
    if (storedToken && storedUser) {
      try {
        setToken(storedToken);
        setUser(JSON.parse(storedUser));
      } catch {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
      }
    }
    setIsLoading(false);
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const response: LoginResponse = await apiLogin(username, password);
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem(AUTH_TOKEN_KEY, response.token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(response.user));
  }, []);

  const register = useCallback(async (username: string, email: string, password: string) => {
    await apiRegister(username, email, password);
    const response: LoginResponse = await apiLogin(username, password);
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem(AUTH_TOKEN_KEY, response.token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(response.user));
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
