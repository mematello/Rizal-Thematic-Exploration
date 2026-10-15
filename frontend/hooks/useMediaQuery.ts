"use client";

import { useSyncExternalStore } from "react";

export function useMediaQuery(query: string): boolean {
  const subscribe = (onStoreChange: () => void) => {
    if (typeof window === "undefined") return () => {};
    const mediaQueryList = window.matchMedia(query);
    mediaQueryList.addEventListener("change", onStoreChange);
    return () => mediaQueryList.removeEventListener("change", onStoreChange);
  };

  const getSnapshot = () => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  };

  const getServerSnapshot = () => false;

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
