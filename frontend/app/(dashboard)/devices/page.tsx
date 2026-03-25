'use client';

import { useEffect, useState, useCallback } from 'react';
import { useAuth } from '@/lib/auth-context';
import {
  listDevices,
  claimDevice,
  updateDevice,
  formatDeviceTypeLabel,
  type Device,
  ApiError,
} from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { FieldGroup, Field, FieldLabel } from '@/components/ui/field';
import { Spinner } from '@/components/ui/spinner';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { toast } from 'sonner';
import {
  Cpu,
  Zap,
  Activity,
  Plus,
  CheckCircle,
  XCircle,
  Battery,
  MapPin,
  Save,
} from 'lucide-react';

function DeviceTypeIcon({ type }: { type: Device['device_type'] }) {
  switch (type) {
    case 'smart_hub':
      return <Cpu className="h-5 w-5 text-primary" />;
    case 'smart_plug':
      return <Zap className="h-5 w-5 text-chart-3" />;
    case 'pir_sensor':
      return <Activity className="h-5 w-5 text-chart-2" />;
    default:
      return <Cpu className="h-5 w-5 text-muted-foreground" />;
  }
}

function DeviceCard({ device, onClick }: { device: Device; onClick: () => void }) {
  return (
    <Card
      className="bg-card border-border cursor-pointer transition-colors hover:border-primary/40"
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-lg bg-secondary flex items-center justify-center">
            <DeviceTypeIcon type={device.device_type} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-medium text-foreground capitalize">
                {formatDeviceTypeLabel(device.device_type)}
              </h3>
              {device.is_active ? (
                <Badge className="bg-primary/20 text-primary border-primary/30">
                  Online
                </Badge>
              ) : (
                <Badge variant="secondary" className="bg-muted text-muted-foreground">
                  Offline
                </Badge>
              )}
            </div>
            <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
              {device.battery_level !== null && (
                <div className="flex items-center gap-1">
                  <Battery className="h-3 w-3" />
                  <span>{device.battery_level}%</span>
                </div>
              )}
              {device.location && (
                <div className="flex items-center gap-1">
                  <MapPin className="h-3 w-3" />
                  <span className="capitalize">{device.location.replace(/_/g, ' ')}</span>
                </div>
              )}
              {device.is_provisioned && (
                <div className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-primary" />
                  <span>Provisioned</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

const LOCATION_OPTIONS = [
  { value: 'living_room', label: 'Living Room' },
  { value: 'bedroom', label: 'Bedroom' },
  { value: 'kitchen', label: 'Kitchen' },
  { value: 'bathroom', label: 'Bathroom' },
  { value: 'office', label: 'Office' },
  { value: 'garage', label: 'Garage' },
] as const;

const SPECIAL_USE_OPTIONS = [
  { value: 'medicine_cabinet', label: 'Medicine Cabinet' },
] as const;

function formatLocation(location: string | null): string {
  if (!location) return 'Not set';
  return location.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function DeviceDetailDialog({
  device,
  open,
  onOpenChange,
  onSave,
}: {
  device: Device | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (id: string, data: Partial<Pick<Device, 'location' | 'special_use'>>) => Promise<void>;
}) {
  const [location, setLocation] = useState<string>('');
  const [specialUse, setSpecialUse] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (device) {
      setLocation(device.location ?? '');
      setSpecialUse(device.special_use ?? '');
    }
  }, [device]);

  if (!device) return null;

  const hasChanges =
    location !== (device.location ?? '') ||
    specialUse !== (device.special_use ?? '');

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave(device.id, {
        location: location || null,
        special_use: specialUse || null,
      });
      onOpenChange(false);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card border-border max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-card-foreground flex items-center gap-2">
            <DeviceTypeIcon type={device.device_type} />
            <span className="capitalize">{formatDeviceTypeLabel(device.device_type)}</span>
            {device.is_active ? (
              <Badge className="bg-primary/20 text-primary border-primary/30">Online</Badge>
            ) : (
              <Badge variant="secondary" className="bg-muted text-muted-foreground">Offline</Badge>
            )}
          </DialogTitle>
          <DialogDescription className="font-mono text-xs">
            {device.serial_number}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          {/* Read-only info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Status</p>
              <div className="flex items-center gap-1.5">
                {device.is_active ? (
                  <CheckCircle className="h-4 w-4 text-primary" />
                ) : (
                  <XCircle className="h-4 w-4 text-muted-foreground" />
                )}
                <span className="font-medium text-foreground">
                  {device.is_active ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Battery</p>
              <div className="flex items-center gap-1.5">
                <Battery className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium text-foreground">
                  {device.battery_level !== null ? `${device.battery_level}%` : 'N/A'}
                </span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Provisioned</p>
              <div className="flex items-center gap-1.5">
                {device.is_provisioned ? (
                  <CheckCircle className="h-4 w-4 text-primary" />
                ) : (
                  <XCircle className="h-4 w-4 text-muted-foreground" />
                )}
                <span className="font-medium text-foreground">
                  {device.is_provisioned ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>

          {/* Editable fields */}
          <div className="space-y-4 pt-2 border-t border-border">
            <h4 className="text-sm font-medium text-foreground">Settings</h4>

            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Location</Label>
              <Select value={location} onValueChange={setLocation}>
                <SelectTrigger className="bg-input border-border">
                  <SelectValue placeholder="Select a location" />
                </SelectTrigger>
                <SelectContent>
                  {LOCATION_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Special Use</Label>
              <Select value={specialUse || '_none'} onValueChange={(v) => setSpecialUse(v === '_none' ? '' : v)}>
                <SelectTrigger className="bg-input border-border">
                  <SelectValue placeholder="None" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="_none">None</SelectItem>
                  {SPECIAL_USE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Save */}
          <div className="flex justify-end gap-3 pt-4 border-t border-border">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!hasChanges || isSaving}>
              {isSaving ? <Spinner className="mr-2" /> : <Save className="h-4 w-4 mr-2" />}
              Save Changes
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ClaimDeviceDialog({
  open,
  onOpenChange,
  onClaim,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onClaim: (serial: string) => Promise<void>;
}) {
  const [serial, setSerial] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!serial.trim()) return;

    setIsSubmitting(true);
    try {
      await onClaim(serial.trim());
      setSerial('');
      onOpenChange(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card border-border">
        <DialogHeader>
          <DialogTitle className="text-card-foreground">Claim a Device</DialogTitle>
          <DialogDescription>
            Enter the serial number from your device packaging or admin panel to claim it.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="serial">Serial Number</FieldLabel>
              <Input
                id="serial"
                type="text"
                value={serial}
                onChange={(e) => setSerial(e.target.value)}
                placeholder="e.g., SH-001234-ABCD"
                className="bg-input border-border text-foreground font-mono"
                required
              />
            </Field>
          </FieldGroup>
          <div className="flex justify-end gap-3 mt-6">
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting || !serial.trim()}>
              {isSubmitting ? <Spinner className="mr-2" /> : null}
              Claim Device
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function DevicesPage() {
  const { token } = useAuth();
  const [devices, setDevices] = useState<Device[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [claimDialogOpen, setClaimDialogOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const fetchDevices = async () => {
    if (!token) return;

    try {
      const data = await listDevices(token);
      setDevices(data);
      setError(null);
    } catch (err) {
      console.error('[v0] Error fetching devices:', err);
      setError('Failed to load devices');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDevices();
  }, [token]);

  const handleClaimDevice = async (serial: string) => {
    if (!token) return;

    try {
      const response = await claimDevice(token, serial);
      const kind = formatDeviceTypeLabel(response.device?.device_type ?? null);
      toast.success('Device claimed successfully', {
        description: `${kind} has been added to your account.`,
      });
      await fetchDevices();
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 409) {
          toast.error('Device already claimed', {
            description: 'This device is already registered to another user.',
          });
        } else {
          toast.error('Failed to claim device', {
            description: err.message,
          });
        }
      } else {
        toast.error('Failed to claim device', {
          description: err instanceof Error ? err.message : 'An unexpected error occurred',
        });
      }
    }
  };

  const openDetail = useCallback((device: Device) => {
    setSelectedDevice(device);
    setDetailOpen(true);
  }, []);

  const handleSaveDevice = async (
    id: string,
    data: Partial<Pick<Device, 'location' | 'special_use'>>,
  ) => {
    if (!token) return;
    try {
      const updated = await updateDevice(token, id, data);
      setDevices((prev) => prev.map((d) => (d.id === id ? updated : d)));
      setSelectedDevice(updated);
      toast.success('Device updated');
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error('Failed to update device', { description: err.message });
      } else {
        toast.error('Failed to update device');
      }
      throw err;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-32 mb-2" />
            <Skeleton className="h-4 w-48" />
          </div>
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="bg-card border-border">
              <CardContent className="p-4">
                <div className="flex items-start gap-4">
                  <Skeleton className="h-12 w-12 rounded-lg" />
                  <div className="flex-1">
                    <Skeleton className="h-5 w-24 mb-2" />
                    <Skeleton className="h-3 w-32 mb-2" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const hubs = devices.filter((d) => d.device_type === 'smart_hub');
  const sensors = devices.filter((d) => d.device_type === 'pir_sensor');
  const plugs = devices.filter((d) => d.device_type === 'smart_plug');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Devices</h1>
          <p className="text-muted-foreground">
            Manage and monitor your connected devices
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-border overflow-hidden">
            <button
              onClick={() => setViewMode('cards')}
              className={`px-3 py-1.5 text-sm ${
                viewMode === 'cards'
                  ? 'bg-secondary text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Cards
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`px-3 py-1.5 text-sm ${
                viewMode === 'table'
                  ? 'bg-secondary text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Table
            </button>
          </div>
          <Button onClick={() => setClaimDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Claim Device
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{hubs.length}</p>
              <p className="text-sm text-muted-foreground">Smart Hubs</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="h-10 w-10 rounded-lg bg-chart-2/10 flex items-center justify-center">
              <Activity className="h-5 w-5 text-chart-2" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{sensors.length}</p>
              <p className="text-sm text-muted-foreground">PIR Sensors</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
              <Zap className="h-5 w-5 text-chart-3" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{plugs.length}</p>
              <p className="text-sm text-muted-foreground">Smart Plugs</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Device List */}
      {devices.length === 0 ? (
        <Card className="bg-card border-border">
          <CardContent className="py-16 text-center">
            <Cpu className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No devices yet</h3>
            <p className="text-muted-foreground mb-4 max-w-md mx-auto">
              Get started by claiming your first device using the serial number from your device packaging.
            </p>
            <Button onClick={() => setClaimDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Claim Your First Device
            </Button>
          </CardContent>
        </Card>
      ) : viewMode === 'cards' ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {devices.map((device) => (
            <DeviceCard key={device.id} device={device} onClick={() => openDetail(device)} />
          ))}
        </div>
      ) : (
        <Card className="bg-card border-border">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Type</TableHead>
                <TableHead className="text-muted-foreground">Serial Number</TableHead>
                <TableHead className="text-muted-foreground">Status</TableHead>
                <TableHead className="text-muted-foreground">Battery</TableHead>
                <TableHead className="text-muted-foreground">Location</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {devices.map((device) => (
                <TableRow
                  key={device.id}
                  className="border-border cursor-pointer"
                  onClick={() => openDetail(device)}
                >
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <DeviceTypeIcon type={device.device_type} />
                      <span className="capitalize text-foreground">
                        {formatDeviceTypeLabel(device.device_type)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-sm text-muted-foreground">
                    {device.serial_number}
                  </TableCell>
                  <TableCell>
                    {device.is_active ? (
                      <div className="flex items-center gap-1.5 text-primary">
                        <CheckCircle className="h-4 w-4" />
                        <span>Online</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 text-muted-foreground">
                        <XCircle className="h-4 w-4" />
                        <span>Offline</span>
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {device.battery_level !== null ? `${device.battery_level}%` : '—'}
                  </TableCell>
                  <TableCell className="text-muted-foreground capitalize">
                    {device.location ? device.location.replace(/_/g, ' ') : '—'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      <ClaimDeviceDialog
        open={claimDialogOpen}
        onOpenChange={setClaimDialogOpen}
        onClaim={handleClaimDevice}
      />

      <DeviceDetailDialog
        device={selectedDevice}
        open={detailOpen}
        onOpenChange={setDetailOpen}
        onSave={handleSaveDevice}
      />
    </div>
  );
}
