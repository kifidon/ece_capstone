import { clearAuthStorageAndRedirectToLogin } from './auth-storage';

/**
 * Next.js injects NEXT_PUBLIC_* at build time. For local dev, use `.env.development` or `.env.local`.
 * If the value has no scheme (e.g. "localhost:8000"), we prepend http:// so fetch() works.
 * Without a base URL, relative `/api/...` requests hit the Next.js origin, not Django — so we default
 * to the local backend in development only.
 */
function normalizeApiBaseUrl(): string {
  let raw = '';
  try {
    raw = (process.env.NEXT_PUBLIC_API_BASE_URL ?? '').trim();
  } catch {
    raw = '';
  }
  if (!raw) {
    raw = 'http://127.0.0.1:8000';
  }
  raw = raw.replace(/\/+$/, '');
  if (!/^https?:\/\//i.test(raw)) {
    raw = `http://${raw.replace(/^\/+/, '')}`;
  }
  return raw;
}

const API_BASE_URL = normalizeApiBaseUrl();

/**
 * Backend URL layout (see Django `api/urls.py`):
 * - `/api/hub/*` — provisioning (claim by serial, hub-only flows live on the device)
 * - `/api/devices/*`, `/api/events/*` — DRF ViewSets (list/detail by id); do not use `/api/devices/claim/` etc.
 */
const routes = {
  auth: {
    register: '/api/auth/register/',
    login: '/api/auth/login/',
    updateProfile: '/api/auth/update_profile/',
  },
  hub: {
    claim: '/api/hub/claim/',
  },
  devices: {
    list: '/api/devices/',
    detail: (id: string) => `/api/devices/${encodeURIComponent(id)}/`,
  },
  events: {
    list: '/api/events/',
    detail: (id: string) => `/api/events/${encodeURIComponent(id)}/`,
  },
} as const;

export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
}

export interface LoginResponse {
  token: string;
  refresh: string;
  user: User;
}

export interface Device {
  id: string;
  /** Backend allows null until the row is fully configured */
  device_type: 'smart_hub' | 'pir_sensor' | 'smart_plug' | null;
  serial_number: string;
  hub_device: string | null;
  user: number;
  battery_level: number | null;
  is_active: boolean;
  location: string | null;
  special_use: string | null;
  ip_address: string | null;
  is_provisioned: boolean;
}

/** Safe label for UI; avoids crashing when API returns null device_type */
export function formatDeviceTypeLabel(type: Device['device_type']): string {
  if (!type || typeof type !== 'string') return 'Device';
  return type.replace(/_/g, ' ');
}

export interface ClaimResponse {
  status: string;
  /** Present when claiming a smart_hub; backend sends `"scheduled"` when config push is queued */
  config_push?: string;
  device: Device;
}

export interface Event {
  id: string;
  timestamp: string;
  action: 'tv' | 'medicine' | 'reaching' | 'unknown';
  pose_classification: 'lying' | 'reaching' | 'sitting' | 'standing' | 'unknown';
  is_alert: boolean;
  is_resolved: boolean;
  is_processed: boolean;
  /** Hub snapshot: usually a JSON array of device objects */
  device_state: unknown;
  trigger_device: unknown;
  /** Populated when `is_alert` and Gemini post-process flagged the event */
  alert_reasoning?: string | null;
  keypoints: unknown;
  inference_result: unknown;
}

class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data: unknown) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

type HandleResponseOptions = {
  /** false for login — 401 is wrong password, not an expired session */
  redirectOnUnauthorized?: boolean;
};

async function handleResponse<T>(
  response: Response,
  options: HandleResponseOptions = {}
): Promise<T> {
  const { redirectOnUnauthorized = true } = options;
  if (!response.ok) {
    if (response.status === 401 && redirectOnUnauthorized) {
      clearAuthStorageAndRedirectToLogin();
    }
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { detail: 'An error occurred' };
    }
    throw new ApiError(
      errorData.detail || errorData.error || 'Request failed',
      response.status,
      errorData
    );
  }
  return response.json();
}

/** DRF list endpoints return a JSON array unless pagination is enabled (`{ results: [...] }`). */
async function handleListResponse<T>(
  response: Response,
  options: HandleResponseOptions = {}
): Promise<T[]> {
  const data = await handleResponse<unknown>(response, options);
  if (Array.isArray(data)) return data as T[];
  if (
    data !== null &&
    typeof data === 'object' &&
    Array.isArray((data as { results?: unknown }).results)
  ) {
    return (data as { results: T[] }).results;
  }
  throw new ApiError('Unexpected API list response shape', response.status, data);
}

export async function register(
  username: string,
  email: string,
  password: string
): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}${routes.auth.register}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  });
  return handleResponse<{ message: string }>(response, { redirectOnUnauthorized: false });
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const response = await fetch(`${API_BASE_URL}${routes.auth.login}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  return handleResponse<LoginResponse>(response, { redirectOnUnauthorized: false });
}

export async function updateProfile(
  token: string,
  data: {
    address?: string;
    phone_number?: string;
    first_name?: string;
    last_name?: string;
    kasa_username?: string;
    kasa_password?: string;
  }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${routes.auth.updateProfile}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

export async function listDevices(token: string): Promise<Device[]> {
  const response = await fetch(`${API_BASE_URL}${routes.devices.list}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleListResponse<Device>(response);
}

export async function getDevice(token: string, id: string): Promise<Device> {
  const response = await fetch(`${API_BASE_URL}${routes.devices.detail(id)}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Device>(response);
}

export async function claimDevice(token: string, serial_number: string): Promise<ClaimResponse> {
  const response = await fetch(`${API_BASE_URL}${routes.hub.claim}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ serial_number }),
  });
  return handleResponse<ClaimResponse>(response);
}

export async function listEvents(token: string): Promise<Event[]> {
  const response = await fetch(`${API_BASE_URL}${routes.events.list}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleListResponse<Event>(response);
}

export async function getEvent(token: string, id: string): Promise<Event> {
  const response = await fetch(`${API_BASE_URL}${routes.events.detail(id)}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Event>(response);
}

export async function deleteEvent(token: string, id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${routes.events.detail(id)}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    if (response.status === 401) {
      clearAuthStorageAndRedirectToLogin();
    }
    throw new ApiError('Failed to delete event', response.status, null);
  }
}

export { ApiError };
