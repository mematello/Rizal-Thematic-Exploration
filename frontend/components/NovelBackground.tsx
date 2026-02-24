"use client";

import { motion } from "framer-motion";

interface NovelBackgroundProps {
    novel: "noli" | "fili" | "both";
}

export function NovelBackground({ novel }: NovelBackgroundProps) {
    // Determine if we are in "fili" mode (dark/revolutionary)
    const isFili = novel === "fili";

    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none select-none" aria-hidden="true">
            {/* Dynamic Backdrop color transition */}
            <motion.div
                className="absolute inset-0 bg-black"
                initial={false}
                animate={{ opacity: isFili ? 0.12 : 0.05 }}
                transition={{ duration: 1.5 }}
            />

            {/* Thematic Background Image Layer */}
            <motion.div
                className="absolute inset-0 z-0 bg-no-repeat bg-cover bg-center transition-all duration-1000"
                initial={false}
                animate={{
                    opacity: novel === "both" ? 0.3 : 0.7,
                    scale: 1.05,
                    filter: "blur(4px) brightness(0.9) contrast(1.1)",
                }}
                style={{
                    backgroundImage: novel === "noli" ? "var(--image-bg-noli)" : novel === "fili" ? "var(--image-bg-fili)" : "var(--image-bg-neutral)",
                }}
            />

            {/* Gradient wash to blend image with UI */}
            <div
                className="absolute inset-0 z-1"
                style={{
                    background: isFili
                        ? "linear-gradient(to bottom, transparent 0%, var(--theme-bg) 80%)"
                        : "linear-gradient(to bottom, transparent 0%, var(--theme-bg) 90%)",
                }}
            />

            {/* Subtle vignette overlay */}
            <div
                className="absolute inset-0 z-2"
                style={{
                    background: isFili
                        ? "radial-gradient(ellipse at center, transparent 30%, rgba(44, 24, 16, 0.4) 100%)"
                        : "radial-gradient(ellipse at center, transparent 40%, rgba(25, 20, 15, 0.25) 100%)",
                }}
            />

            {/* Gradient sky layers — warm gold to deep navy */}
            <motion.div
                className="absolute inset-0 z-3"
                animate={{
                    filter: isFili ? "brightness(0.9) contrast(1.1) saturate(0.8)" : "brightness(1) contrast(1) saturate(1)",
                }}
                transition={{ duration: 1.5 }}
                style={{
                    background: isFili ? `
            radial-gradient(ellipse 80% 60% at 50% -10%, rgba(197,160,101,0.22) 0%, transparent 70%),
            radial-gradient(ellipse 60% 50% at 80% 20%, rgba(141,45,45,0.15) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 10% 30%, rgba(27,38,59,0.12) 0%, transparent 55%),
            linear-gradient(to bottom, rgba(141,45,45,0.05) 0%, transparent 60%)
          ` : `
            radial-gradient(ellipse 90% 70% at 50% -15%, rgba(255,255,255,0.9) 0%, rgba(252,250,247,0.4) 40%, transparent 80%),
            radial-gradient(ellipse 60% 50% at 80% 20%, rgba(255,255,255,0.3) 0%, transparent 60%),
            linear-gradient(to bottom, rgba(255,255,255,0.5) 0%, transparent 50%)
          `,
                }}
            />

            {/* Main SVG illustration */}
            <svg
                viewBox="0 0 1440 560"
                preserveAspectRatio="xMidYMid slice"
                className="absolute inset-0 w-full h-full"
                xmlns="http://www.w3.org/2000/svg"
            >
                <defs>
                    {/* Gold shimmer gradient */}
                    <linearGradient id="goldSky" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={isFili ? "#C5A065" : "#FFFFFF"} stopOpacity={isFili ? "0.18" : "0.6"} />
                        <stop offset="100%" stopColor="#1B263B" stopOpacity="0.05" />
                    </linearGradient>

                    {/* Church silhouette gradient */}
                    <linearGradient id="churchGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#1B263B" stopOpacity="0.18" />
                        <stop offset="100%" stopColor="#1B263B" stopOpacity="0.28" />
                    </linearGradient>

                    {/* Glowing sun gradient */}
                    <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#C5A065" stopOpacity="0.55" />
                        <stop offset="50%" stopColor="#C5A065" stopOpacity="0.18" />
                        <stop offset="100%" stopColor="#C5A065" stopOpacity="0" />
                    </radialGradient>

                    {/* Crimson fili glow */}
                    <radialGradient id="filiGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#8D2D2D" stopOpacity="0.30" />
                        <stop offset="100%" stopColor="#8D2D2D" stopOpacity="0" />
                    </radialGradient>

                    {/* Soft blur for glow effects */}
                    <filter id="softGlow">
                        <feGaussianBlur stdDeviation="8" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Sky gradient wash */}
                <rect width="1440" height="560" fill="url(#goldSky)" />

                {/* ===== PHILIPPINE SUN — center-left background ===== */}
                <motion.g
                    transform="translate(260, 180)"
                    opacity={isFili ? 0.08 : 0.12} // Slightly dimmer in Fili mode
                    filter="url(#softGlow)"
                    animate={{ opacity: isFili ? 0.08 : 0.12 }}
                    transition={{ duration: 1.5 }}
                >
                    {/* Outer glow halo */}
                    <circle r="110" fill="url(#sunGlow)" />
                    {/* Sun circle */}
                    <circle r="44" fill="#C5A065" opacity="0.90" />
                    {/* 8 rays */}
                    {Array.from({ length: 8 }).map((_, i) => {
                        const angle = (i * 45 * Math.PI) / 180;
                        const x1 = Math.cos(angle) * 52;
                        const y1 = Math.sin(angle) * 52;
                        const x2 = Math.cos(angle) * 96;
                        const y2 = Math.sin(angle) * 96;
                        return (
                            <line
                                key={i}
                                x1={x1} y1={y1}
                                x2={x2} y2={y2}
                                stroke="#C5A065"
                                strokeWidth={i % 2 === 0 ? "5" : "3"}
                                strokeLinecap="round"
                            />
                        );
                    })}
                </motion.g>

                {/* ===== CRIMSON GLOW — top right (El Filibusterismo) ===== */}
                <motion.ellipse
                    cx="1280" cy="80" rx="220" ry="160"
                    fill="url(#filiGlow)"
                    animate={{ opacity: isFili ? 1 : 0.4 }} // Stronger red glow in Fili
                    transition={{ duration: 1.5 }}
                />

                {/* ===== COLONIAL CHURCH SILHOUETTE — left ===== */}
                <g opacity="0.13" fill="url(#churchGrad)">
                    {/* Church body */}
                    <rect x="60" y="300" width="160" height="180" rx="2" />
                    {/* Bell tower left */}
                    <rect x="65" y="230" width="50" height="90" rx="2" />
                    {/* Bell tower right */}
                    <rect x="165" y="245" width="50" height="75" rx="2" />
                    {/* Tower caps */}
                    <polygon points="65,230 90,190 115,230" />
                    <polygon points="165,245 190,205 215,245" />
                    {/* Cross on left tower */}
                    <rect x="86" y="182" width="4" height="18" />
                    <rect x="81" y="188" width="14" height="4" />
                    {/* Cross on right tower */}
                    <rect x="188" y="197" width="4" height="16" />
                    <rect x="183" y="202" width="14" height="4" />
                    {/* Arched door */}
                    <path d="M115 480 L115 390 Q130 370 145 390 L145 480 Z" />
                    {/* Rose window */}
                    <circle cx="130" cy="330" r="16" fill="none" stroke="#1B263B" strokeWidth="3" opacity="0.5" />
                    <circle cx="130" cy="330" r="8" fill="#1B263B" opacity="0.3" />
                </g>

                {/* ===== BAHAY NA BATO SILHOUETTE — far right ===== */}
                <g transform="translate(1180, 0)" opacity="0.10" fill="#1B263B">
                    {/* Main house */}
                    <rect x="100" y="340" width="210" height="150" rx="2" />
                    {/* Roof */}
                    <polygon points="85,340 205,260 325,340" />
                    {/* Windows row */}
                    <rect x="130" y="370" width="30" height="40" rx="3" />
                    <rect x="180" y="370" width="30" height="40" rx="3" />
                    <rect x="240" y="370" width="30" height="40" rx="3" />
                    {/* Door */}
                    <rect x="185" y="430" width="40" height="60" rx="3" />
                </g>

                {/* ===== ROLLING HILLS / GROUND SILHOUETTE ===== */}
                <path
                    d="M0 480 Q200 430 400 460 Q600 490 800 450 Q1000 415 1200 445 Q1350 465 1440 440 L1440 560 L0 560 Z"
                    fill="#1B263B"
                    opacity="0.08"
                />

                {/* ===== NARRA TREE — left ===== */}
                <g transform="translate(380, 0)" opacity="0.10" fill="#1B263B">
                    {/* Trunk */}
                    <rect x="28" y="380" width="14" height="100" rx="4" />
                    {/* Canopy layers */}
                    <ellipse cx="35" cy="340" rx="55" ry="60" />
                    <ellipse cx="35" cy="310" rx="42" ry="48" />
                    <ellipse cx="35" cy="285" rx="30" ry="36" />
                </g>

                {/* ===== NARRA TREE — right ===== */}
                <g transform="translate(1080, 40)" opacity="0.08" fill="#1B263B">
                    <rect x="28" y="340" width="12" height="90" rx="4" />
                    <ellipse cx="34" cy="305" rx="48" ry="55" />
                    <ellipse cx="34" cy="280" rx="36" ry="42" />
                    <ellipse cx="34" cy="260" rx="24" ry="30" />
                </g>

                {/* ===== BAMBOO STALKS — right side ===== */}
                <g transform="translate(1310, 200)" opacity="0.09" stroke="#1B263B" strokeWidth="5" fill="none">
                    <line x1="0" y1="0" x2="-10" y2="360" strokeWidth="6" />
                    <line x1="20" y1="20" x2="15" y2="360" strokeWidth="5" />
                    <line x1="45" y1="5" x2="50" y2="360" strokeWidth="4" />
                    <line x1="70" y1="30" x2="65" y2="360" strokeWidth="5" />
                    {/* Nodes */}
                    {[80, 160, 240, 320].map((y) => (
                        <g key={y}>
                            <line x1="-15" y1={y} x2="5" y2={y} strokeWidth="3" />
                            <line x1="12" y1={y + 15} x2="28" y2={y + 15} strokeWidth="2.5" />
                            <line x1="40" y1={y + 5} x2="58" y2={y + 5} strokeWidth="2.5" />
                            <line x1="62" y1={y + 20} x2="78" y2={y + 20} strokeWidth="2" />
                        </g>
                    ))}
                </g>

                {/* ===== SAMPAGUITA FLOWERS scattered ===== */}
                {[
                    { cx: 900, cy: 80, r: 6 },
                    { cx: 940, cy: 55, r: 5 },
                    { cx: 870, cy: 110, r: 4 },
                    { cx: 960, cy: 95, r: 5 },
                    { cx: 920, cy: 130, r: 4 },
                ].map((f, i) => (
                    <g key={i} opacity="0.18">
                        {/* 5-petal flower */}
                        {Array.from({ length: 5 }).map((_, p) => {
                            const angle = (p * 72 * Math.PI) / 180;
                            return (
                                <ellipse
                                    key={p}
                                    cx={f.cx + Math.cos(angle) * f.r}
                                    cy={f.cy + Math.sin(angle) * f.r}
                                    rx={f.r * 0.8}
                                    ry={f.r * 1.4}
                                    transform={`rotate(${p * 72}, ${f.cx + Math.cos(angle) * f.r}, ${f.cy + Math.sin(angle) * f.r})`}
                                    fill="#C5A065"
                                />
                            );
                        })}
                        <circle cx={f.cx} cy={f.cy} r={f.r * 0.6} fill="#FCFAF7" opacity="0.8" />
                    </g>
                ))}

                {/* ===== ORNAMENTAL CORNER FILIGREE — top left ===== */}
                <g opacity="0.12" stroke="#C5A065" strokeWidth="1.2" fill="none">
                    <path d="M0,0 Q40,0 40,40" transform="translate(10,10)" />
                    <path d="M0,0 Q60,0 60,60" transform="translate(10,10)" />
                    <path d="M0,0 Q80,0 80,80" transform="translate(10,10)" />
                    <path d="M10,10 L10,50 Q10,80 50,80" />
                    <circle cx="50" cy="10" r="3" fill="#C5A065" />
                    <circle cx="10" cy="50" r="3" fill="#C5A065" />
                </g>

                {/* ===== ORNAMENTAL CORNER FILIGREE — top right ===== */}
                <g opacity="0.12" stroke="#C5A065" strokeWidth="1.2" fill="none" transform="translate(1440,0) scale(-1,1)">
                    <path d="M0,0 Q40,0 40,40" transform="translate(10,10)" />
                    <path d="M0,0 Q60,0 60,60" transform="translate(10,10)" />
                    <path d="M0,0 Q80,0 80,80" transform="translate(10,10)" />
                    <path d="M10,10 L10,50 Q10,80 50,80" />
                    <circle cx="50" cy="10" r="3" fill="#C5A065" />
                    <circle cx="10" cy="50" r="3" fill="#C5A065" />
                </g>

                {/* ===== QUILL PEN — subtle center ===== */}
                <g transform="translate(700, 60)" opacity="0.07" fill="#1B263B">
                    {/* Feather */}
                    <path d="M40 0 Q80 60 20 140 Q0 100 30 60 Q50 40 40 0Z" />
                    <path d="M40 0 Q10 50 15 130 Q20 100 35 70 Q45 40 40 0Z" fill="#C5A065" />
                    {/* Quill shaft */}
                    <line x1="30" y1="60" x2="22" y2="155" stroke="#1B263B" strokeWidth="2" />
                    {/* Nib */}
                    <path d="M20 150 L16 170 L24 166 Z" fill="#1B263B" />
                </g>

                {/* ===== HORIZONTAL DIVIDER LINE WITH ORNAMENT ===== */}
                <g opacity="0.08">
                    <line x1="100" y1="555" x2="660" y2="555" stroke="#C5A065" strokeWidth="0.8" />
                    <line x1="780" y1="555" x2="1340" y2="555" stroke="#C5A065" strokeWidth="0.8" />
                    <circle cx="720" cy="555" r="5" fill="#C5A065" />
                    <circle cx="705" cy="555" r="3" fill="#C5A065" opacity="0.6" />
                    <circle cx="735" cy="555" r="3" fill="#C5A065" opacity="0.6" />
                </g>
            </svg>

            {/* Animated floating particles (dust motes / bokeh) */}
            <div className="absolute inset-0">
                {[
                    { x: "15%", y: "20%", size: 3, delay: 0 },
                    { x: "75%", y: "15%", size: 2, delay: 1.2 },
                    { x: "45%", y: "35%", size: 4, delay: 2.5 },
                    { x: "85%", y: "50%", size: 2, delay: 0.8 },
                    { x: "25%", y: "60%", size: 3, delay: 3.1 },
                    { x: "60%", y: "75%", size: 2, delay: 1.7 },
                    { x: "90%", y: "25%", size: 2, delay: 4.0 },
                    { x: "5%", y: "80%", size: 3, delay: 2.0 },
                ].map((p, i) => (
                    <motion.div
                        key={i}
                        className="absolute rounded-full bg-brand-gold"
                        style={{
                            left: p.x,
                            top: p.y,
                            width: p.size,
                            height: p.size,
                            opacity: 0,
                        }}
                        animate={{
                            opacity: isFili ? [0, 0.45, 0] : [0, 0.35, 0], // Slightly brighter dust in dark mode
                            y: [0, -24, -48],
                        }}
                        transition={{
                            duration: 6,
                            repeat: Infinity,
                            delay: p.delay,
                            ease: "easeInOut",
                        }}
                    />
                ))}
            </div>
        </div>
    );
}
