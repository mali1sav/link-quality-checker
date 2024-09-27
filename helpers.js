"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isRelevant = exports.formatRelevancyResult = void 0;
function formatRelevancyResult(result) {
    return `Relevancy Score: ${result.score}/100\n\nExplanation: ${result.explanation}`;
}
exports.formatRelevancyResult = formatRelevancyResult;
function isRelevant(score, threshold = 70) {
    return score >= threshold;
}
exports.isRelevant = isRelevant;
