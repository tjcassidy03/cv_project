from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

class CRAFTTextDetector:
    def __init__(self, image_path, output_dir, text_threshold=0.7, link_threshold=0.4, low_text=0.3, cuda=False, long_size=1280, boundary_type = "polys"):
        self.image_path = image_path
        self.output_dir = output_dir
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.boundary_type = boundary_type

        self.image = read_image(self.image_path)
        self.craft_net = load_craftnet_model(cuda=self.cuda)
        self.refine_net = load_refinenet_model(cuda=self.cuda)

    def run(self):
        prediction_result = get_prediction(
            image=self.image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.cuda,
            long_size=self.long_size
        )

        if self.boundary_type != "polys" and self.boundary_type != "boxes":
            print("invalid boundary type!")
            return 

        export_detected_regions(
            image=self.image,
            regions=prediction_result[self.boundary_type],
            output_dir=self.output_dir,
            rectify=True
        )

        export_extra_results(
            image=self.image,
            regions=prediction_result[self.boundary_type],
            heatmaps=prediction_result["heatmaps"],
            output_dir=self.output_dir
        )

        # empty_cuda_cache()

