from docextractorfork import Extractor
from docextractorfork.extractor import BACKGROUND_LABEL, LABEL_TO_NAME, ILLUSTRATION_LABEL



class IlluExtractor():
    def __init__(self):
        self.extractor = Extractor(input_dir, args.output_dir, labels_to_extract=args.labels, tag=args.tag,
                          save_annotations=args.save_annot, straight_bbox=args.straight_bbox,
                          draw_margin=args.draw_margin)
    def run(self):
        for filename in self.extractor.files:
            self.extractor.print_and_log_info('Processing {}'.format(filename.relative_to(self.extractor.input_dir)))
            try:
                imgs_with_names, output_path = self.extractor.get_images_and_output_path(filename)
            except (NotImplementedError, OSError) as e:
                self.extractor.print_and_log_error(e)
                imgs_with_names, output_path = [], None

            for img, name in imgs_with_names:
                img_with_annotations = img.copy()
                pred = self.extractor.predict(img)
                for label in self.extractor.labels_to_extract:
                    if label != BACKGROUND_LABEL:
                        extracted_elements = self.extractor.extract(img, pred, label, img_with_annotations)
                        path = output_path if len(self.extractor.labels_to_extract) == 1 else output_path / LABEL_TO_NAME[label]
                        for k, extracted_element in enumerate(extracted_elements):
                            extracted_element.save(path / '{}_{}.{}'
                                                   .format(name, k, self.extractor.out_extension))
                if self.extractor.save_annotations:
                    (output_path / 'annotation').mkdir(exist_ok=True)
                    img_with_annotations.save(output_path / 'annotation' / '{}_annotated.{}'
                                              .format(name, self.extractor.out_extension))

        self.extractor.print_and_log_info('Extractor run is over')