"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { User } from "lucide-react";

interface CharacterAvatarProps {
    name: string;
    className?: string; // For container styling
    size?: number; // Pixel size for the image/icon
    priority?: boolean;
    onClick?: () => void;
}

export function CharacterAvatar({ name, className = "", size = 80, priority = false, onClick }: CharacterAvatarProps) {
    const [src, setSrc] = useState<string>("");
    const [hasError, setHasError] = useState(false);

    useEffect(() => {
        // Simple slugify: lowercase, replace spaces with hyphens, remove special chars
        const slug = name.toLowerCase()
            .replace(/[.\.]/g, '') // remove dots (e.g. Dr., Mr.)
            .replace(/ñ/g, 'n') // normalize ñ
            .replace(/[^a-z0-9\s-]/g, '') // remove other special chars
            .trim()
            .replace(/\s+/g, '-');

        setSrc(`/images/characters/${slug}.png`);
        setHasError(false);
    }, [name]);

    if (hasError || !src) {
        return (
            <div
                className={`flex items-center justify-center bg-brand-cream border-2 border-brand-gold/10 rounded-full text-brand-navy opacity-80 ${className}`}
                style={{ width: size, height: size }}
            >
                <User size={size * 0.5} />
            </div>
        );
    }

    return (
        <div
            onClick={onClick}
            className={`relative overflow-hidden rounded-full bg-brand-cream border-2 border-brand-gold/10 ${onClick ? 'cursor-pointer hover:opacity-90 transition-opacity' : ''} ${className}`}
            style={{ width: size, height: size }}
        >
            <Image
                src={src}
                alt={name}
                fill
                className="object-cover"
                onError={() => setHasError(true)}
                sizes={`${size}px`}
                priority={priority}
            />
        </div>
    );
}
