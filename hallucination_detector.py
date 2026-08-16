from sentence_transformers import CrossEncoder

from config import NLI_MODEL


class HallucinationDetector:

    def __init__(self):

        print("Loading NLI model...")

        self.model = CrossEncoder(
            NLI_MODEL
        )

    def evaluate(self, answer, context):

        sentences = [
            sentence.strip()
            for sentence in answer.split(".")
            if sentence.strip()
        ]

        if not sentences:
            return {
                "score": 0.0,
                "label": "UNKNOWN"
            }

        scores = []

        for sentence in sentences:

            prediction = self.model.predict(
                [(context, sentence)]
            )

            score = float(prediction[0])

            scores.append(score)

        average_score = sum(scores) / len(scores)

        if average_score >= 0.70:

            label = "SUPPORTED"

        elif average_score >= 0.40:

            label = "UNCERTAIN"

        else:

            label = "POSSIBLE_HALLUCINATION"

        return {
            "score": round(average_score, 4),
            "label": label
        }