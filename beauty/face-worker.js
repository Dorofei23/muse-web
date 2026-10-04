/* global Vision, importScripts */
importScripts("./vision_bundle.js");
let detector;
self.onmessage = async ({ data }) => {
  const { image } = data;
  try {
    detector ??= Vision.FilesetResolver.forVisionTasks(
      new URL("./wasm", self.location.href).href,
    ).then((files) =>
      Vision.FaceLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: new URL("./face_landmarker.task", self.location.href)
            .href,
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        numFaces: 2,
      }),
    );
    const result = (await detector).detect(image);
    let hairMask;
    if (result.faceLandmarks.length === 1) {
      // Hair failure must not prevent the independent face makeup controls.
      try {
        const files = await Vision.FilesetResolver.forVisionTasks(
          new URL("./wasm", self.location.href).href,
        );
        const segmenter = await Vision.ImageSegmenter.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: new URL(
              "./hair_segmenter.tflite",
              self.location.href,
            ).href,
            delegate: "CPU",
          },
          runningMode: "IMAGE",
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        });
        try {
          const index = segmenter
            .getLabels()
            .findIndex((label) => label.toLowerCase().includes("hair"));
          segmenter.segment(image, (segmentation) => {
            const mask = segmentation.confidenceMasks?.[index];
            if (mask) {
              const values = new Float32Array(mask.getAsFloat32Array());
              if (values.some((value) => value > 0.65))
                hairMask = { width: mask.width, height: mask.height, values };
            }
          });
        } finally {
          segmenter.close();
        }
      } catch {
        /* Leave makeup usable when hair segmentation is unavailable. */
      }
    }
    self.postMessage(
      { landmarks: result.faceLandmarks, hairMask },
      hairMask ? [hairMask.values.buffer] : [],
    );
  } catch {
    detector = undefined;
    self.postMessage({
      error:
        "Не удалось загрузить примерку. Проверь соединение и попробуй снова.",
    });
  } finally {
    image.close();
  }
};
