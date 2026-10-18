import type { Metadata } from "next";
import {
  NotifyPage,
  type NotifySearchParams,
} from "@/components/notify/NotifyPage";

// Public (no sign-in, REC-134). Opening it changes nothing; the button POSTs.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Confirm emails · WebAgent",
  referrer: "no-referrer",
  robots: { index: false, follow: false },
};

export default async function ConfirmPage({
  searchParams,
}: {
  searchParams: NotifySearchParams;
}) {
  return <NotifyPage purpose="confirm" searchParams={searchParams} />;
}
