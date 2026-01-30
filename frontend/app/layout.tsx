import type { Metadata } from "next";
import { Roboto, Crimson_Text } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const roboto = Roboto({
  variable: "--font-roboto",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const crimson = Crimson_Text({
  variable: "--font-crimson",
  subsets: ["latin"],
  weight: ["400", "600", "700"],
});

export const metadata: Metadata = {
  title: "Rizal Thematic Exploration",
  description: "Semantic search for Noli Me Tangere and El Filibusterismo",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${roboto.variable} ${crimson.variable} antialiased bg-brand-cream text-brand-text`}
      >
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
