'use client'

import * as React from 'react'
import { ThemeProvider } from '@/components/theme-provider'

/**
 * Locks UI to light theme so system dark mode / Sonner don't flip the app to grey/dark.
 */
export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      forcedTheme="light"
    >
      {children}
    </ThemeProvider>
  )
}
