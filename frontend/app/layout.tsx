import type { Metadata } from "next";
import { DM_Serif_Display, Lora, Figtree } from "next/font/google";
import "./globals.css";

const dmSerifDisplay = DM_Serif_Display({ subsets: ["latin"], weight: "400", variable: "--font-display" });
const lora = Lora({ subsets: ["latin"], variable: "--font-lora" });
const figtree = Figtree({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "NeuroTalk — Voice Intelligence Platform",
  description: "Live transcription, AI reasoning, and expressive voice synthesis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${dmSerifDisplay.variable} ${lora.variable} ${figtree.variable}`}>{children}</body>
    </html>
  );
}
