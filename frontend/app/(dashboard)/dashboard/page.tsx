'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { listDevices, listEvents, type Device, type Event } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Cpu,
  AlertTriangle,
  Activity,
  CheckCircle,
  Clock,
  Zap,
} from 'lucide-react';
import Link from 'next/link';

function StatCard({
  title,
  value,
  description,
  icon: Icon,
  trend,
  href,
}: {
  title: string;
  value: string | number;
  description: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'neutral';
  href?: string;
}) {
  const content = (
    <Card className="bg-card border-border hover:border-primary/50 transition-colors">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
          <Icon className="h-4 w-4 text-primary" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold text-card-foreground">{value}</div>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  );

  if (href) {
    return <Link href={href}>{content}</Link>;
  }

  return content;
}

function RecentEventCard({ event }: { event: Event }) {
  const getActionColor = (action: string) => {
    switch (action) {
      case 'medicine':
        return 'bg-chart-2 text-foreground';
      case 'reaching':
        return 'bg-chart-3 text-foreground';
      case 'tv':
        return 'bg-chart-5 text-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getPoseIcon = (pose: string) => {
    switch (pose) {
      case 'lying':
        return '🛏';
      case 'standing':
        return '🧍';
      case 'sitting':
        return '🪑';
      case 'reaching':
        return '🙆';
      default:
        return '❓';
    }
  };

  return (
    <Link
      href={`/events?event=${encodeURIComponent(event.id)}`}
      className="flex items-center gap-4 p-3 rounded-lg bg-secondary/50 border border-border hover:border-primary/60 hover:bg-secondary transition-colors cursor-pointer"
    >
      <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center text-lg">
        {getPoseIcon(event.pose_classification)}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium text-foreground capitalize">
            {event.action} Activity
          </p>
          {event.is_alert && (
            <Badge variant="destructive" className="text-xs">
              Alert
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {new Date(event.timestamp).toLocaleString()}
        </p>
      </div>
      <Badge className={getActionColor(event.action)}>
        {event.pose_classification}
      </Badge>
    </Link>
  );
}

export default function DashboardPage() {
  const { token, user } = useAuth();
  const [devices, setDevices] = useState<Device[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      if (!token) return;

      try {
        const [devicesData, eventsData] = await Promise.all([
          listDevices(token),
          listEvents(token),
        ]);
        setDevices(devicesData);
        setEvents(eventsData);
      } catch (err) {
        console.error('[v0] Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, [token]);

  const onlineDevices = devices.filter((d) => d.is_active).length;
  const alerts24h = events.filter((e) => {
    const eventDate = new Date(e.timestamp);
    const now = new Date();
    const diff = now.getTime() - eventDate.getTime();
    return e.is_alert && diff < 24 * 60 * 60 * 1000;
  }).length;
  const unresolvedAlerts = events.filter((e) => e.is_alert && !e.is_resolved).length;
  const recentEvents = events.slice(0, 5);

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div>
          <Skeleton className="h-8 w-64 mb-2" />
          <Skeleton className="h-4 w-96" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="bg-card border-border">
              <CardHeader className="pb-2">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16 mb-2" />
                <Skeleton className="h-3 w-32" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">
          Welcome back, {user?.first_name || user?.username}
        </h1>
        <p className="text-muted-foreground">
          {"Here's an overview of your monitoring system"}
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Devices Online"
          value={onlineDevices}
          description={`${devices.length} total devices`}
          icon={Cpu}
          href="/devices"
        />
        <StatCard
          title="Alerts (24h)"
          value={alerts24h}
          description="Events requiring attention"
          icon={AlertTriangle}
          href="/events"
        />
        <StatCard
          title="Unresolved"
          value={unresolvedAlerts}
          description="Alerts awaiting resolution"
          icon={Clock}
          href="/events"
        />
        <StatCard
          title="Total Events"
          value={events.length}
          description="All recorded events"
          icon={Activity}
          href="/events"
        />
      </div>

      {/* Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent Events */}
        <Card className="bg-card border-border">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-card-foreground">Recent Events</CardTitle>
                <CardDescription>Latest activity from your devices</CardDescription>
              </div>
              <Link
                href="/events"
                className="text-sm text-primary hover:underline"
              >
                View all
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            {recentEvents.length === 0 ? (
              <p className="text-muted-foreground text-sm">No events recorded yet</p>
            ) : (
              <div className="space-y-3">
                {recentEvents.map((event) => (
                  <RecentEventCard key={event.id} event={event} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Device Status */}
        <Card className="bg-card border-border">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-card-foreground">Device Status</CardTitle>
                <CardDescription>Current state of your devices</CardDescription>
              </div>
              <Link
                href="/devices"
                className="text-sm text-primary hover:underline"
              >
                Manage
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            {devices.length === 0 ? (
              <div className="text-center py-8">
                <Cpu className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground text-sm mb-2">No devices claimed yet</p>
                <Link
                  href="/devices"
                  className="text-sm text-primary hover:underline"
                >
                  Claim your first device
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {devices.slice(0, 5).map((device) => (
                  <div
                    key={device.id}
                    className="flex items-center gap-4 p-3 rounded-lg bg-secondary/50 border border-border"
                  >
                    <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center">
                      {device.device_type === 'smart_hub' ? (
                        <Cpu className="h-5 w-5 text-primary" />
                      ) : device.device_type === 'smart_plug' ? (
                        <Zap className="h-5 w-5 text-chart-3" />
                      ) : (
                        <Activity className="h-5 w-5 text-chart-2" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground capitalize">
                        {device.device_type.replace('_', ' ')}
                      </p>
                      <p className="text-xs text-muted-foreground font-mono">
                        {device.serial_number}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {device.battery_level !== null && (
                        <span className="text-xs text-muted-foreground">
                          {device.battery_level}%
                        </span>
                      )}
                      {device.is_active ? (
                        <CheckCircle className="h-4 w-4 text-primary" />
                      ) : (
                        <div className="h-2 w-2 rounded-full bg-muted-foreground" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
