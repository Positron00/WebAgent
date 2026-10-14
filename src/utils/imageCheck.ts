/** Checks for images sent to /api/chat (IMP-061). Shared by the route and the browser. */

export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
export const IMAGE_TOO_LARGE = "Image is larger than 5 MB";

const IMAGE_RE = /^data:image\/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)$/;

/** Returns the reason the image is rejected, or null when it is fine (or absent). */
export function imageError(image: unknown): string | null {
  if (image == null) return null;
  if (typeof image !== "string") return "Image must be a data URL";
  const match = IMAGE_RE.exec(image);
  if (!match) return "Only PNG, JPEG, GIF or WebP images are accepted";
  const base64 = match[2].replace(/=+$/, "");
  const decodedBytes = Math.floor((base64.length * 3) / 4);
  if (decodedBytes > MAX_IMAGE_BYTES) return IMAGE_TOO_LARGE;
  return null;
}
