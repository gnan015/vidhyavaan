class RAGEvaluator:

    def __init__(self, threshold=1.2):

        self.threshold = threshold

    def evaluate(self, results):

        distances = results.get("distances", [[]])[0]

        if not distances:
            return {
                "relevant": False,
                "score": None
            }

        best_distance = distances[0]

        relevant = best_distance <= self.threshold

        return {
            "relevant": relevant,
            "score": best_distance
        }