import type { Metadata, Viewport } from "next";
import { Playfair_Display, Cormorant_Garamond } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";

const playfair = Playfair_Display({
  variable: "--font-serif",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const cormorant = Cormorant_Garamond({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export const metadata: Metadata = {
  title: "NoliElFili | Buod, Buong Kwento at Sanggunian ni Rizal",
  description: "Tuklasin ang Noli Me Tangere at El Filibusterismo: Buod, Buong Kwento, mga tauhan at tema. Hanapin ang mga siping tumutugma sa Buod gamit ang Sanggunian.",
  openGraph: {
    title: "NoliElFili | Buod, Buong Kwento at Sanggunian ni Rizal",
    description: "Tuklasin ang Noli Me Tangere at El Filibusterismo: Buod, Buong Kwento, mga tauhan at tema. Hanapin ang mga siping tumutugma sa Buod gamit ang Sanggunian.",
    url: "https://nolielfili.com/",
    siteName: "NoliElFili", type: "website", locale: "fil_PH",
    images: [{ url: "https://nolielfili.com/icon.png", width: 512, height: 512, alt: "NoliElFili" }],
  },
  twitter: {
    card: "summary",
    title: "NoliElFili | Buod, Buong Kwento at Sanggunian ni Rizal",
    description: "Tuklasin ang Noli Me Tangere at El Filibusterismo: Buod, Buong Kwento, mga tauhan at tema. Hanapin ang mga siping tumutugma sa Buod gamit ang Sanggunian.",
    images: ["https://nolielfili.com/icon.png"],
  },
  manifest: "/site.webmanifest",
  icons: {
    icon: [
      { url: "/favicon.ico", type: "image/x-icon", sizes: "16x16 32x32 48x48 64x64 128x128 256x256" },
      ...[16, 32, 48, 192, 512].map((size) => ({
        url: `/favicon-${size}x${size}.png`, type: "image/png", sizes: `${size}x${size}`,
      })),
    ],
    apple: [{ url: "/apple-icon.png", sizes: "180x180", type: "image/png" }],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tl" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                var stored = localStorage.getItem('rizal-novel-store');
                if (stored) {
                  var parsed = JSON.parse(stored);
                  var n = parsed && parsed.state && parsed.state.selectedNovel;
                  if (n === 'fili' || n === 'noli') {
                    document.documentElement.setAttribute('data-novel', n);
                  }
                }
              } catch (e) {}
            `,
          }}
        />
      </head>
      <body
        className={`${playfair.variable} ${cormorant.variable} antialiased bg-brand-cream text-brand-text min-h-[100dvh] flex flex-col justify-between w-full max-w-full overflow-x-clip`}
        suppressHydrationWarning
      >
        <Providers>
          {children}
        </Providers>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
