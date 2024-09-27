interface RelevancyResult {
    score: number;
    explanation: string;
}

export function formatRelevancyResult(result: RelevancyResult): string {
    return `Relevancy Score: ${result.score}/100\n\nExplanation: ${result.explanation}`;
}

export function isRelevant(score: number, threshold: number = 70): boolean {
    return score >= threshold;
}