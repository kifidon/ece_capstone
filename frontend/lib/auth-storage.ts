/** Keep in sync with AuthProvider localStorage keys. */
export const AUTH_TOKEN_KEY = 'auth_token';
export const AUTH_USER_KEY = 'auth_user';

export function clearAuthStorageAndRedirectToLogin(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
  window.location.href = '/login';
}
