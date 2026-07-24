from .coco_evaluator import COCOEvaluator as Evaluator

evaluators = {
    "coco": Evaluator,
    "yolo": Evaluator,
}
