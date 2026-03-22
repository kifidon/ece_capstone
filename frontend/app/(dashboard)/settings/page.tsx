'use client';

import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { updateProfile, ApiError } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { FieldGroup, Field, FieldLabel } from '@/components/ui/field';
import { Spinner } from '@/components/ui/spinner';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import {
  User,
  Phone,
  MapPin,
  Zap,
  CheckCircle,
  Eye,
  EyeOff,
} from 'lucide-react';

function ProfileSection() {
  const { user, token } = useAuth();
  const [firstName, setFirstName] = useState(user?.first_name || '');
  const [lastName, setLastName] = useState(user?.last_name || '');
  const [address, setAddress] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;

    setIsSubmitting(true);
    try {
      await updateProfile(token, {
        first_name: firstName,
        last_name: lastName,
        address: address || undefined,
        phone_number: phoneNumber || undefined,
      });
      toast.success('Profile updated successfully');
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error('Failed to update profile', { description: err.message });
      } else {
        toast.error('Failed to update profile');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <User className="h-5 w-5 text-primary" />
          </div>
          <div>
            <CardTitle className="text-card-foreground">Profile Information</CardTitle>
            <CardDescription>Update your personal details</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit}>
          <FieldGroup>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field>
                <FieldLabel htmlFor="firstName">First Name</FieldLabel>
                <Input
                  id="firstName"
                  type="text"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  placeholder="John"
                  className="bg-input border-border text-foreground"
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="lastName">Last Name</FieldLabel>
                <Input
                  id="lastName"
                  type="text"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  placeholder="Doe"
                  className="bg-input border-border text-foreground"
                />
              </Field>
            </div>
            <Field>
              <FieldLabel htmlFor="address">
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Address
                </div>
              </FieldLabel>
              <Input
                id="address"
                type="text"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="123 Main St, City, State"
                className="bg-input border-border text-foreground"
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="phoneNumber">
                <div className="flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  Phone Number
                </div>
              </FieldLabel>
              <Input
                id="phoneNumber"
                type="tel"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="+1 (555) 123-4567"
                className="bg-input border-border text-foreground"
              />
            </Field>
          </FieldGroup>

          <Button type="submit" className="mt-6" disabled={isSubmitting}>
            {isSubmitting ? <Spinner className="mr-2" /> : null}
            Save Changes
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function KasaCredentialsSection() {
  const { token } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;

    setIsSubmitting(true);
    try {
      await updateProfile(token, {
        kasa_username: username,
        kasa_password: password,
      });
      setIsSaved(true);
      setPassword('');
      toast.success('Kasa credentials saved', {
        description: 'Your hub will use these credentials for Kasa/Tapo cloud control.',
      });
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error('Failed to save Kasa credentials', { description: err.message });
      } else {
        toast.error('Failed to save Kasa credentials');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
            <Zap className="h-5 w-5 text-chart-3" />
          </div>
          <div>
            <CardTitle className="text-card-foreground">Kasa / Tapo Credentials</CardTitle>
            <CardDescription>
              Connect your TP-Link Kasa or Tapo account for smart plug control
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isSaved ? (
          <div className="flex items-center gap-3 p-4 rounded-lg bg-primary/10 border border-primary/20">
            <CheckCircle className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium text-foreground">Credentials Saved</p>
              <p className="text-sm text-muted-foreground">
                Your Kasa credentials are securely stored. The hub will use them for smart plug control.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="ml-auto"
              onClick={() => setIsSaved(false)}
            >
              Update
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="kasaUsername">Kasa/Tapo Email</FieldLabel>
                <Input
                  id="kasaUsername"
                  type="email"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="your-email@example.com"
                  className="bg-input border-border text-foreground"
                  required
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="kasaPassword">Kasa/Tapo Password</FieldLabel>
                <div className="relative">
                  <Input
                    id="kasaPassword"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="bg-input border-border text-foreground pr-10"
                    required
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Your password is encrypted and never displayed after saving.
                </p>
              </Field>
            </FieldGroup>

            <Button type="submit" className="mt-6" disabled={isSubmitting}>
              {isSubmitting ? <Spinner className="mr-2" /> : null}
              Save Credentials
            </Button>
          </form>
        )}
      </CardContent>
    </Card>
  );
}

function AccountSection() {
  const { user } = useAuth();

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle className="text-card-foreground">Account Information</CardTitle>
        <CardDescription>Your account details (read-only)</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg bg-secondary">
            <div>
              <p className="text-xs text-muted-foreground">Username</p>
              <p className="text-sm font-medium text-foreground">{user?.username}</p>
            </div>
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-secondary">
            <div>
              <p className="text-xs text-muted-foreground">Email</p>
              <p className="text-sm font-medium text-foreground">{user?.email}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function SettingsPage() {
  return (
    <div className="space-y-6 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground">
          Manage your account and integrations
        </p>
      </div>

      <ProfileSection />

      <Separator className="bg-border" />

      <KasaCredentialsSection />

      <Separator className="bg-border" />

      <AccountSection />
    </div>
  );
}
