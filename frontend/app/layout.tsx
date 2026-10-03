import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TacticEye",
  description: "Football tracking and tactical analytics dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
