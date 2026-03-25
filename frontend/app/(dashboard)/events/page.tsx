'use client';

import { Suspense, useEffect, useState, useMemo, useRef } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { listEvents, deleteEvent, resolveEvent, type Event, ApiError } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { toast } from 'sonner';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  AlertReasoningPanel,
  DeviceStateSection,
  TriggerDeviceSection,
} from '@/components/events/event-context-display';
import {
  AlertTriangle,
  CheckCircle,
  Clock,
  Filter,
  Activity,
  Trash2,
  X,
} from 'lucide-react';

type ActionFilter = 'all' | 'tv' | 'medicine' | 'reaching' | 'unknown';
type PoseFilter = 'all' | 'lying' | 'reaching' | 'sitting' | 'standing' | 'unknown';

/** Resolution applies to alerts only; non-alerts are neither resolved nor unresolved. */
function isUnresolvedAlert(event: Event): boolean {
  return event.is_alert && !event.is_resolved;
}

function minutesUntilNextHour(): number {
  return 60 - new Date().getMinutes();
}

function EventDetailDialog({
  event,
  open,
  onOpenChange,
  onDelete,
  onResolve,
}: {
  event: Event | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: (id: string) => void;
  onResolve: (id: string) => void;
}) {
  const [isDeleting, setIsDeleting] = useState(false);

  if (!event) return null;

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      onDelete(event.id);
      onOpenChange(false);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card border-border max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-card-foreground flex items-center gap-2">
            Event Details
            {event.is_alert && (
              <Badge variant="destructive">Alert</Badge>
            )}
          </DialogTitle>
          <DialogDescription>
            {new Date(event.timestamp).toLocaleString()}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          <AlertReasoningPanel event={event} />

          {/* Summary */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Action</p>
              <p className="font-medium text-foreground capitalize">{event.action}</p>
            </div>
            <div className="p-3 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Pose</p>
              <p className="font-medium text-foreground capitalize">{event.pose_classification}</p>
            </div>
          </div>

          {/* Status */}
          <div className="flex flex-wrap gap-2">
            {!event.is_processed ? (
              <Badge variant="outline" className="text-muted-foreground font-normal">
                <Clock className="h-3 w-3 mr-1" />
                Processing in ~{minutesUntilNextHour()}m
              </Badge>
            ) : !event.is_alert ? (
              <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30">
                <CheckCircle className="h-3 w-3 mr-1" />
                Normal Activity
              </Badge>
            ) : event.is_resolved ? (
              <Badge className="bg-primary/20 text-primary border-primary/30">
                <CheckCircle className="h-3 w-3 mr-1" />
                Resolved
              </Badge>
            ) : (
              <Badge variant="destructive">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Alert
              </Badge>
            )}
          </div>

          <DeviceStateSection deviceState={event.device_state} />
          <TriggerDeviceSection
            triggerDevice={event.trigger_device}
            deviceState={event.device_state}
          />

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-border">
            {event.is_alert && !event.is_resolved && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      onClick={() => { onResolve(event.id); onOpenChange(false); }}
                    >
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Not an issue
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs text-center">
                    <p>Mark this alert as safe. This helps improve future alert detection.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={isDeleting}
            >
              Delete
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function EventsPageInner() {
  const { token } = useAuth();
  const searchParams = useSearchParams();
  const router = useRouter();
  const eventParam = searchParams.get('event');
  const openedFromQuery = useRef<string | null>(null);
  const prevEventParam = useRef<string | null>(null);

  useEffect(() => {
    if (eventParam !== prevEventParam.current) {
      openedFromQuery.current = null;
      prevEventParam.current = eventParam;
    }
  }, [eventParam]);

  const [events, setEvents] = useState<Event[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  // Filters
  const [alertOnly, setAlertOnly] = useState(false);
  const [unresolvedOnly, setUnresolvedOnly] = useState(false);
  const [actionFilter, setActionFilter] = useState<ActionFilter>('all');
  const [poseFilter, setPoseFilter] = useState<PoseFilter>('all');
  const [showFilters, setShowFilters] = useState(false);

  const fetchEvents = async () => {
    if (!token) return;

    try {
      const data = await listEvents(token);
      // Sort by timestamp descending
      data.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setEvents(data);
      setError(null);
    } catch (err) {
      console.error('[v0] Error fetching events:', err);
      setError('Failed to load events');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [token]);

  useEffect(() => {
    if (!eventParam || isLoading || events.length === 0) return;
    if (openedFromQuery.current === eventParam) return;
    const ev = events.find((e) => e.id === eventParam);
    if (ev) {
      openedFromQuery.current = eventParam;
      setSelectedEvent(ev);
      setDetailOpen(true);
      router.replace('/events', { scroll: false });
    }
  }, [eventParam, events, isLoading, router]);

  const filteredEvents = useMemo(() => {
    return events.filter((event) => {
      if (alertOnly && !event.is_alert) return false;
      if (unresolvedOnly && !isUnresolvedAlert(event)) return false;
      if (actionFilter !== 'all' && event.action !== actionFilter) return false;
      if (poseFilter !== 'all' && event.pose_classification !== poseFilter) return false;
      return true;
    });
  }, [events, alertOnly, unresolvedOnly, actionFilter, poseFilter]);

  const handleDeleteEvent = async (id: string) => {
    if (!token) return;
    try {
      await deleteEvent(token, id);
      setEvents((prev) => prev.filter((e) => e.id !== id));
      toast.success('Event deleted');
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error('Failed to delete event', { description: err.message });
      } else {
        toast.error('Failed to delete event');
      }
    }
  };

  const handleResolveEvent = async (id: string) => {
    if (!token) return;
    try {
      const updated = await resolveEvent(token, id);
      setEvents((prev) => prev.map((e) => (e.id === id ? updated : e)));
      toast.success('Alert marked as not an issue');
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error('Failed to resolve event', { description: err.message });
      } else {
        toast.error('Failed to resolve event');
      }
    }
  };

  const handleViewEvent = (event: Event) => {
    setSelectedEvent(event);
    setDetailOpen(true);
  };

  const clearFilters = () => {
    setAlertOnly(false);
    setUnresolvedOnly(false);
    setActionFilter('all');
    setPoseFilter('all');
  };

  const hasActiveFilters = alertOnly || unresolvedOnly || actionFilter !== 'all' || poseFilter !== 'all';

  const alertCount = events.filter((e) => e.is_alert).length;
  const unresolvedCount = events.filter(isUnresolvedAlert).length;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-32 mb-2" />
            <Skeleton className="h-4 w-48" />
          </div>
        </div>
        <Card className="bg-card border-border">
          <CardContent className="p-0">
            <div className="space-y-0">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex items-center gap-4 p-4 border-b border-border last:border-0">
                  <Skeleton className="h-10 w-10 rounded-lg" />
                  <div className="flex-1">
                    <Skeleton className="h-4 w-32 mb-2" />
                    <Skeleton className="h-3 w-48" />
                  </div>
                  <Skeleton className="h-6 w-20" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Events</h1>
          <p className="text-muted-foreground">
            {events.length} total events, {alertCount} alerts, {unresolvedCount} unresolved alert
            {unresolvedCount === 1 ? '' : 's'}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => setShowFilters(!showFilters)}
          className={hasActiveFilters ? 'border-primary text-primary' : ''}
        >
          <Filter className="h-4 w-4 mr-2" />
          Filters
          {hasActiveFilters && (
            <Badge className="ml-2 bg-primary text-primary-foreground">
              {[alertOnly, unresolvedOnly, actionFilter !== 'all', poseFilter !== 'all'].filter(Boolean).length}
            </Badge>
          )}
        </Button>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {/* Filters Panel */}
      {showFilters && (
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-6">
              {/* Alert Only */}
              <div className="flex items-center gap-2">
                <Switch
                  id="alert-only"
                  checked={alertOnly}
                  onCheckedChange={setAlertOnly}
                />
                <Label htmlFor="alert-only" className="text-sm">Alerts only</Label>
              </div>

              {/* Unresolved Only */}
              <div className="flex items-center gap-2">
                <Switch
                  id="unresolved-only"
                  checked={unresolvedOnly}
                  onCheckedChange={setUnresolvedOnly}
                />
                <Label htmlFor="unresolved-only" className="text-sm">
                  Unresolved alerts only
                </Label>
              </div>

              {/* Action Filter */}
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground">Action:</Label>
                <Select value={actionFilter} onValueChange={(v) => setActionFilter(v as ActionFilter)}>
                  <SelectTrigger className="w-32 bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="tv">TV</SelectItem>
                    <SelectItem value="medicine">Medicine</SelectItem>
                    <SelectItem value="reaching">Reaching</SelectItem>
                    <SelectItem value="unknown">Unknown</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Pose Filter */}
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground">Pose:</Label>
                <Select value={poseFilter} onValueChange={(v) => setPoseFilter(v as PoseFilter)}>
                  <SelectTrigger className="w-32 bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="lying">Lying</SelectItem>
                    <SelectItem value="sitting">Sitting</SelectItem>
                    <SelectItem value="standing">Standing</SelectItem>
                    <SelectItem value="reaching">Reaching</SelectItem>
                    <SelectItem value="unknown">Unknown</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Clear Filters */}
              {hasActiveFilters && (
                <Button variant="ghost" size="sm" onClick={clearFilters}>
                  <X className="h-4 w-4 mr-1" />
                  Clear
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Events List */}
      {filteredEvents.length === 0 ? (
        <Card className="bg-card border-border">
          <CardContent className="py-16 text-center">
            <Activity className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              {events.length === 0 ? 'No events yet' : 'No events match your filters'}
            </h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              {events.length === 0
                ? 'Events will appear here as your devices record activity.'
                : 'Try adjusting your filter settings to see more events.'}
            </p>
            {hasActiveFilters && (
              <Button variant="outline" className="mt-4" onClick={clearFilters}>
                Clear Filters
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card className="bg-card border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Time</TableHead>
                <TableHead className="text-muted-foreground">Action</TableHead>
                <TableHead className="text-muted-foreground">Pose</TableHead>
                <TableHead className="text-muted-foreground">Alert status</TableHead>
                <TableHead className="text-muted-foreground text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEvents.map((event) => (
                <TableRow
                  key={event.id}
                  className="border-border cursor-pointer hover:bg-secondary/50"
                  onClick={() => handleViewEvent(event)}
                >
                  <TableCell>
                    <div>
                      <p className="text-sm text-foreground">
                        {new Date(event.timestamp).toLocaleDateString()}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="text-foreground capitalize">{event.action}</span>
                      {event.is_alert && (
                        <AlertTriangle className="h-4 w-4 text-destructive" />
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="capitalize">
                      {event.pose_classification}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {!event.is_processed ? (
                        <Badge variant="outline" className="text-muted-foreground font-normal">
                          <Clock className="h-3 w-3 mr-1" />
                          Processing in ~{minutesUntilNextHour()}m
                        </Badge>
                      ) : !event.is_alert ? (
                        <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Normal Activity
                        </Badge>
                      ) : event.is_resolved ? (
                        <Badge className="bg-primary/20 text-primary border-primary/30">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Resolved
                        </Badge>
                      ) : (
                        <Badge variant="destructive">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          Alert
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1">
                      {event.is_alert && !event.is_resolved && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleResolveEvent(event.id);
                                }}
                              >
                                <CheckCircle className="h-4 w-4 text-muted-foreground hover:text-primary" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-xs text-center">
                              <p>Mark as safe — helps improve future alerts</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteEvent(event.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      <EventDetailDialog
        event={selectedEvent}
        open={detailOpen}
        onOpenChange={setDetailOpen}
        onDelete={handleDeleteEvent}
        onResolve={handleResolveEvent}
      />
    </div>
  );
}

function EventsPageFallback() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Skeleton className="h-8 w-32 mb-2" />
          <Skeleton className="h-4 w-48" />
        </div>
      </div>
      <Card className="bg-card border-border">
        <CardContent className="p-8 flex justify-center">
          <Skeleton className="h-10 w-48" />
        </CardContent>
      </Card>
    </div>
  );
}

export default function EventsPage() {
  return (
    <Suspense fallback={<EventsPageFallback />}>
      <EventsPageInner />
    </Suspense>
  );
}
