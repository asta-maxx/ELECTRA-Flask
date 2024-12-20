import { useLocation } from 'react-router-dom';

function ResultsVisualization() {
    const location = useLocation();
    const results = location.state?.results;

    if (!results) {
        return <div>No Results Available</div>;
    }

    return (
        <div>
            <h1>Training Results</h1>
            <div>
                <p><strong>Status:</strong> {results.status}</p>
                <p><strong>Accuracy:</strong> {results.final_accuracy}</p>
                <p><strong>Predictions:</strong></p>
                <pre>{JSON.stringify(results.predictions, null, 2)}</pre>
                <p><strong>Actuals:</strong></p>
                <pre>{JSON.stringify(results.actuals, null, 2)}</pre>
                <p><strong>Model Archive:</strong> <a href={`/${results.model_archive}`} target="_blank" rel="noopener noreferrer">Download Model</a></p>
            </div>
        </div>
    );
}


export default ResultsVisualization;