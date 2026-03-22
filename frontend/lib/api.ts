/**
 * Next.js injects NEXT_PUBLIC_* at build time. For local dev, use `.env.development` or `.env.local`.
 * If the value has no scheme (e.g. "localhost:8000"), we prepend http:// so fetch() works.
 */
function normalizeApiBaseUrl(): string {
  let raw = (process.env.NEXT_PUBLIC_API_BASE_URL || '').trim();
  if (!raw) return '';
  raw = raw.replace(/\/+$/, '');
  if (!/^https?:\/\//i.test(raw)) {
    raw = `http://${raw.replace(/^\/+/, '')}`;
  }
  return raw;
}

const API_BASE_URL = normalizeApiBaseUrl();

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
  device_type: 'smart_hub' | 'pir_sensor' | 'smart_plug';
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

export interface ClaimResponse {
  status: string;
  config_push?: boolean;
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
  device_state: Record<string, unknown>;
  trigger_device: Record<string, unknown>;
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

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
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

export async function register(
  username: string,
  email: string,
  password: string
): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/auth/register/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  });
  return handleResponse<{ message: string }>(response);
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const response = await fetch(`${API_BASE_URL}/api/auth/login/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  return handleResponse<LoginResponse>(response);
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
  const response = await fetch(`${API_BASE_URL}/api/auth/update_profile/`, {
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
  const response = await fetch(`${API_BASE_URL}/api/devices/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Device[]>(response);
}

export async function getDevice(token: string, id: string): Promise<Device> {
  const response = await fetch(`${API_BASE_URL}/api/devices/${id}/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Device>(response);
}

export async function claimDevice(token: string, serial_number: string): Promise<ClaimResponse> {
  const response = await fetch(`${API_BASE_URL}/api/devices/claim/`, {
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
  const response = await fetch(`${API_BASE_URL}/api/events/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Event[]>(response);
}

export async function getEvent(token: string, id: string): Promise<Event> {
  const response = await fetch(`${API_BASE_URL}/api/events/${id}/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse<Event>(response);
}

export async function deleteEvent(token: string, id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/events/${id}/`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new ApiError('Failed to delete event', response.status, null);
  }
}

export { ApiError };
