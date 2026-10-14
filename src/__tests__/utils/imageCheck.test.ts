import { imageError, MAX_IMAGE_BYTES } from "@/utils/imageCheck";

const dataUrl = (bytes: number, type = "png") =>
  `data:image/${type};base64,${Buffer.alloc(bytes).toString("base64")}`;

describe("imageError (IMP-061)", () => {
  it("accepts no image", () => {
    expect(imageError(null)).toBeNull();
    expect(imageError(undefined)).toBeNull();
  });

  it.each(["png", "jpeg", "gif", "webp"])("accepts a %s data URL", (type) => {
    expect(imageError(dataUrl(10, type))).toBeNull();
  });

  it("accepts exactly 5 MB, matching the browser's file size check", () => {
    expect(imageError(dataUrl(MAX_IMAGE_BYTES))).toBeNull();
  });

  it("rejects one byte over 5 MB", () => {
    expect(imageError(dataUrl(MAX_IMAGE_BYTES + 1))).toBe(
      "Image is larger than 5 MB",
    );
  });
});
