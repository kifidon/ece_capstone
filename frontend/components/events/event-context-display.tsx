'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import type { Event } from '@/lib/api';
import { AlertTriangle, Bot, Plug, Radio, ScanLine } from 'lucide-react';

function asDeviceRecords(raw: unknown): Record<string, unknown>[] {
  if (Array.isArray(raw)) {
    return raw.filter((x): x is Record<string, unknown> => x !== null && typeof x === 'object');
  }
  if (raw !== null && typeof raw === 'object' && !Array.isArray(raw)) {
    return [raw as Record<string, unknown>];
  }
  return [];
}

function formatLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatTriggerTimestamp(value: unknown): string | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  if (typeof value === 'number' && Number.isFinite(value)) {
    const ms = value > 1e12 ? value : value * 1000;
    return new Date(ms).toLocaleString();
  }
  if (typeof value === 'string') {
    const n = Number(value);
    if (!Number.isNaN(n) && /^-?\d+(\.\d+)?$/.test(value.trim())) {
      const ms = n > 1e12 ? n : n * 1000;
      return new Date(ms).toLocaleString();
    }
    return value;
  }
  return String(value);
}

function DeviceTypeIcon({ type }: { type: string }) {
  const t = type.toLowerCase();
  if (t === 'smart_plug') return <Plug className="h-4 w-4 text-amber-600 dark:text-amber-400" />;
  if (t === 'pir_sensor') return <Radio className="h-4 w-4 text-sky-600 dark:text-sky-400" />;
  return <ScanLine className="h-4 w-4 text-muted-foreground" />;
}

function DeviceFieldRow({ label, value }: { label: string; value: unknown }) {
  if (value === undefined || value === null || value === '') return null;
  const display =
    typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
  return (
    <div className="flex justify-between gap-4 text-sm py-1.5 border-b border-border/60 last:border-0">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="font-medium text-foreground text-right break-all">{display}</span>
    </div>
  );
}

/** Skip keys we show in the header or redundant */
const SKIP_KEYS = new Set([
  'device_type',
  'serial_number',
  'alias',
  'is_on',
  'battery_level',
  'special_use',
  'sensed',
  'last_seen',
]);

export function AlertReasoningPanel({ event }: { event: Event }) {
  if (!event.is_processed) return null;

  const reasoning = event.alert_reasoning?.trim();
  if (!reasoning) return null;

  const isAlert = event.is_alert;

  return (
    <Card
      className={
        isAlert
          ? 'border-amber-500/40 bg-linear-to-br from-amber-500/5 via-card to-card shadow-sm'
          : 'border-emerald-500/40 bg-linear-to-br from-emerald-500/5 via-card to-card shadow-sm'
      }
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2 text-foreground">
          <span
            className={`flex h-8 w-8 items-center justify-center rounded-lg ${
              isAlert ? 'bg-amber-500/15' : 'bg-emerald-500/15'
            }`}
          >
            <Bot
              className={`h-4 w-4 ${
                isAlert
                  ? 'text-amber-700 dark:text-amber-400'
                  : 'text-emerald-700 dark:text-emerald-400'
              }`}
            />
          </span>
          {isAlert ? 'Why this was flagged' : 'Why this is normal'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">{reasoning}</p>
      </CardContent>
    </Card>
  );
}

export function DeviceStateSection({ deviceState }: { deviceState: unknown }) {
  const devices = asDeviceRecords(deviceState);
  if (devices.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-foreground">Devices at time of event</p>
        <Badge variant="secondary" className="font-normal">
          {devices.length} device{devices.length === 1 ? '' : 's'}
        </Badge>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {devices.map((dev, idx) => {
          const dtype = String(dev.device_type ?? 'device');
          const serial = String(dev.serial_number ?? '—');
          const title = dev.alias ? String(dev.alias) : serial;
          return (
            <Card
              key={`${serial}-${idx}`}
              className="bg-secondary/40 border-border overflow-hidden"
            >
              <CardHeader className="py-3 px-4 space-y-0">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 rounded-md bg-background/80 p-2 border border-border">
                    <DeviceTypeIcon type={dtype} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <CardTitle className="text-sm font-semibold leading-tight truncate">
                      {title}
                    </CardTitle>
                    <div className="flex flex-wrap items-center gap-2 mt-1.5">
                      <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
                        {dtype.replace(/_/g, ' ')}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0 px-4 pb-3">
                <DeviceFieldRow label="Power on" value={dev.is_on} />
                <DeviceFieldRow label="Battery" value={dev.battery_level != null ? `${dev.battery_level}%` : undefined} />
                <DeviceFieldRow label="Last seen" value={formatTriggerTimestamp(dev.last_seen)} />
                <DeviceFieldRow label="Special use" value={dev.special_use} />
                <DeviceFieldRow label="Motion sensed" value={dev.sensed} />
                {Object.entries(dev).map(([k, v]) => {
                  if (SKIP_KEYS.has(k) || v === undefined || v === null || v === '') return null;
                  return <DeviceFieldRow key={k} label={formatLabel(k)} value={v} />;
                })}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

export function TriggerDeviceSection({
  triggerDevice,
  deviceState,
}: {
  triggerDevice: unknown;
  deviceState: unknown;
}) {
  if (triggerDevice === null || typeof triggerDevice !== 'object' || Array.isArray(triggerDevice)) {
    return null;
  }
  const t = triggerDevice as Record<string, unknown>;
  if (Object.keys(t).length === 0) return null;

  const serial = t.serial_number != null ? String(t.serial_number) : '';
  const stateDevices = asDeviceRecords(deviceState);
  const stateSerials = new Set(
    stateDevices.map((d) => (d.serial_number != null ? String(d.serial_number) : '')).filter(Boolean)
  );
  const inSyncList = serial ? stateSerials.has(serial) : true;

  const dtype =
    typeof t.device_type === 'string'
      ? t.device_type
      : serial.toUpperCase().includes('PIR')
        ? 'pir_sensor'
        : undefined;

  return (
    <div className="space-y-3">
      <p className="text-sm font-semibold text-foreground">What triggered this event</p>
      <Card className="border-primary/30 bg-primary/5 overflow-hidden">
        <CardContent className="pt-4 pb-4">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-primary/10 p-2.5 border border-primary/20">
              <Radio className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1 min-w-0 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                {dtype && (
                  <Badge className="bg-primary/90 text-primary-foreground uppercase text-[10px] tracking-wide">
                    {dtype.replace(/_/g, ' ')}
                  </Badge>
                )}
              </div>
              <Separator />
              <div className="space-y-0">
                <DeviceFieldRow label="Battery" value={t.battery_level != null ? `${t.battery_level}%` : undefined} />
                <DeviceFieldRow label="Timestamp" value={formatTriggerTimestamp(t.timestamp)} />
                {Object.entries(t).map(([k, v]) => {
                  if (
                    ['device_type', 'serial_number', 'battery_level', 'timestamp', 'sensed'].includes(k)
                  ) {
                    return null;
                  }
                  if (v === undefined || v === null || v === '') return null;
                  return <DeviceFieldRow key={k} label={formatLabel(k)} value={v} />;
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
