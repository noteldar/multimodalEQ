'use client';

import Chart from 'chart.js/auto';
import Image from 'next/image';
import { useEffect, useRef, useState } from 'react';

export default function KhabibPage() {
    const voiceToneChartRef = useRef<HTMLCanvasElement>(null);
    const facialExpressionChartRef = useRef<HTMLCanvasElement>(null);
    const heartRateChartRef = useRef<HTMLCanvasElement>(null);
    const spiderChartVerbalRef = useRef<HTMLCanvasElement>(null);
    const spiderChartNonVerbalRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [videoDuration, setVideoDuration] = useState<number>(9); // Default fallback based on expected video length
    const voiceToneChartInstanceRef = useRef<Chart | null>(null);
    const facialExpressionChartInstanceRef = useRef<Chart | null>(null);
    const heartRateChartInstanceRef = useRef<Chart | null>(null);
    const spiderChartVerbalInstanceRef = useRef<Chart | null>(null);
    const spiderChartNonVerbalInstanceRef = useRef<Chart | null>(null);
    const [facialExpressionData, setFacialExpressionData] = useState<Array<{
        frame: string;
        num_faces: number;
        expressions: { [key: string]: string };
    }> | null>(null);
    const [heartRateData, setHeartRateData] = useState<Array<{
        time_seconds: number;
        heart_rate_bpm: number;
    }> | null>(null);
    const [verbalEmotionData, setVerbalEmotionData] = useState<Array<{
        emotion: string;
        level: number;
    }> | null>(null);
    const [nonverbalEmotionData, setNonverbalEmotionData] = useState<Array<{
        emotion: string;
        level: number;
    }> | null>(null);

    // Load facial expression data
    useEffect(() => {
        fetch('/khabib.json')
            .then(response => response.json())
            .then(data => {
                console.log('Loaded facial expression data:', data.length, 'frames');
                setFacialExpressionData(data);
            })
            .catch(error => console.error('Error loading facial expression data:', error));
    }, []);

    // Load heart rate data
    useEffect(() => {
        fetch('/khabib.csv')
            .then(response => response.text())
            .then(csvText => {
                const lines = csvText.trim().split('\n');
                const data = lines.slice(1).map(line => {
                    const values = line.split(',');
                    return {
                        time_seconds: parseFloat(values[0]),
                        heart_rate_bpm: parseFloat(values[1])
                    };
                });
                console.log('Loaded heart rate data:', data.length, 'points');
                setHeartRateData(data);
            })
            .catch(error => console.error('Error loading heart rate data:', error));
    }, []);

    // Load verbal emotion data
    useEffect(() => {
        fetch('/text_khabib.json')
            .then(response => response.json())
            .then(data => {
                console.log('Loaded verbal emotion data:', data);
                setVerbalEmotionData(data);
            })
            .catch(error => console.error('Error loading verbal emotion data:', error));
    }, []);

    // Load non-verbal emotion data
    useEffect(() => {
        fetch('/nonverbal_khabib.json')
            .then(response => response.json())
            .then(data => {
                console.log('Loaded non-verbal emotion data:', data);
                setNonverbalEmotionData(data);
            })
            .catch(error => console.error('Error loading non-verbal emotion data:', error));
    }, []);

    // Function to process facial expression data for charting
    const processFacialExpressionData = (data: Array<{
        frame: string;
        num_faces: number;
        expressions: { [key: string]: string };
    }>, videoDuration: number): {
        labels: string[];
        expressionCounts: Array<{
            Sadness: number;
            Anger: number;
            Surprise: number;
            Neutral: number;
            Disgust: number;
            Happiness: number;
            Fear: number;
        }>;
        timePoints: number[];
    } => {
        if (!data || data.length === 0) return { labels: [], expressionCounts: [], timePoints: [] };

        // Map frame indices to time points evenly across video duration
        const timePoints = data.map((frame, index) => {
            // Map frames evenly across the video duration
            return (index / (data.length - 1)) * videoDuration;
        });

        // Count expressions across all faces for each frame
        const expressionCounts = data.map(frame => {
            const counts = {
                Sadness: 0,
                Anger: 0,
                Surprise: 0,
                Neutral: 0,
                Disgust: 0,
                Happiness: 0,
                Fear: 0
            };

            Object.values(frame.expressions).forEach((expression: string) => {
                if (counts.hasOwnProperty(expression)) {
                    counts[expression as keyof typeof counts]++;
                }
            });

            // Convert to proportions (0-1)
            const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
            const proportions = Object.keys(counts).reduce((acc, emotion) => {
                acc[emotion as keyof typeof counts] = total > 0 ? counts[emotion as keyof typeof counts] / total : 0;
                return acc;
            }, {} as typeof counts);

            return proportions;
        });

        return { labels: timePoints.map(time => time.toFixed(1)), expressionCounts, timePoints };
    };

    // Function to generate time labels based on video duration
    const generateTimeLabels = (duration: number) => {
        const interval = duration / 3; // 4 points = 3 intervals
        return [
            '0s',
            `${Math.round(interval)}s`,
            `${Math.round(interval * 2)}s`,
            `${Math.round(duration)}s`
        ];
    };

    // Handle video metadata loaded to get duration
    const handleVideoMetadataLoaded = () => {
        if (videoRef.current && !isNaN(videoRef.current.duration)) {
            const duration = videoRef.current.duration;
            console.log('Video duration detected:', duration);
            setVideoDuration(duration);
        }
    };

    // Additional handler for when video data is loaded
    const handleVideoCanPlay = () => {
        if (videoRef.current && !isNaN(videoRef.current.duration)) {
            const duration = videoRef.current.duration;
            console.log('Video can play, duration:', duration);
            setVideoDuration(duration);
        }
    };

    useEffect(() => {
        // Chart configuration variables
        const chartTextColor = '#A0AEC0';
        const chartGridColor = '#4A5568';
        const accentPrimary = '#63B3ED';
        const accentSecondary = '#4FD1C5';

        // Emotion timeline data from the JSON file
        const emotionData = {
            "Anxiety": [0.0233, 0.158, 0.0131, 0.1404],
            "Pain": [0.0107, 0.1524, 0.0035, 0.1183],
            "Sadness": [0.0331, 0.1501, 0.0042, 0.1459],
            "Determination": [0.0333, 0.0471, 0.1606, 0.0805],
            "Concentration": [0.0248, 0.011, 0.303, 0.0214],
            "Calmness": [0.0874, 0.0329, 0.2227, 0.0465],
            "Distress": [0.0422, 0.1429, 0.0093, 0.1858],
            "Boredom": [0.1048, 0.0567, 0.0229, 0.0328],
            "Interest": [0.0629, 0.0045, 0.1213, 0.0389]
        };

        const timeLabels = generateTimeLabels(videoDuration);
        console.log('Generating chart with duration:', videoDuration, 'labels:', timeLabels);

        const defaultLineChartOptions = (yAxisLabel: string, xAxisLabel = 'Time') => ({
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: yAxisLabel,
                        color: chartTextColor,
                        font: {
                            weight: 500,
                            size: 10
                        }
                    },
                    ticks: {
                        color: chartTextColor,
                        font: {
                            size: 9
                        }
                    },
                    grid: {
                        color: chartGridColor
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: xAxisLabel,
                        color: chartTextColor,
                        font: {
                            weight: 500,
                            size: 10
                        }
                    },
                    ticks: {
                        color: chartTextColor,
                        font: {
                            size: 9
                        }
                    },
                    grid: {
                        color: chartGridColor
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom' as const,
                    labels: {
                        color: chartTextColor,
                        usePointStyle: true,
                        boxWidth: 6,
                        padding: 8,
                        font: {
                            size: 9
                        }
                    }
                }
            }
        });

        // Voice/Tone Progression Chart with emotion data
        if (voiceToneChartRef.current) {
            const ctx = voiceToneChartRef.current.getContext('2d');
            if (ctx) {
                // Destroy existing chart if it exists
                if (voiceToneChartInstanceRef.current) {
                    voiceToneChartInstanceRef.current.destroy();
                    voiceToneChartInstanceRef.current = null;
                }

                voiceToneChartInstanceRef.current = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: timeLabels,
                        datasets: [
                            {
                                label: 'Anxiety',
                                data: emotionData.Anxiety,
                                borderColor: '#EF6C6C',
                                backgroundColor: 'rgba(239, 108, 108, 0.2)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: '#EF6C6C',
                                borderDash: [5, 5]
                            },
                            {
                                label: 'Determination',
                                data: emotionData.Determination,
                                borderColor: accentPrimary,
                                backgroundColor: 'rgba(99, 179, 237, 0.2)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: accentPrimary,
                                borderDash: [5, 5]
                            },
                            {
                                label: 'Concentration',
                                data: emotionData.Concentration,
                                borderColor: '#10B981',
                                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: '#10B981',
                                borderDash: [5, 5]
                            },
                            {
                                label: 'Calmness',
                                data: emotionData.Calmness,
                                borderColor: accentSecondary,
                                backgroundColor: 'rgba(79, 209, 197, 0.2)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: accentSecondary,
                                borderDash: [5, 5]
                            },
                            {
                                label: 'Interest',
                                data: emotionData.Interest,
                                borderColor: '#FBBF24',
                                backgroundColor: 'rgba(251, 191, 36, 0.2)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: '#FBBF24',
                                borderDash: [5, 5]
                            }
                        ]
                    },
                    options: defaultLineChartOptions('Emotion Intensity')
                });
            }
        }

        // Facial Expressions Chart with processed data
        if (facialExpressionChartRef.current && facialExpressionData) {
            const ctx = facialExpressionChartRef.current.getContext('2d');
            if (ctx) {
                // Destroy existing chart if it exists
                if (facialExpressionChartInstanceRef.current) {
                    facialExpressionChartInstanceRef.current.destroy();
                    facialExpressionChartInstanceRef.current = null;
                }

                const processedData = processFacialExpressionData(facialExpressionData, videoDuration);

                if (processedData.labels.length > 0 && processedData.expressionCounts.length > 0) {
                    const datasets = [
                        {
                            label: 'Sadness',
                            data: processedData.expressionCounts.map((counts, index) => ({
                                x: processedData.timePoints[index],
                                y: counts.Sadness
                            })),
                            borderColor: '#EF6C6C',
                            backgroundColor: 'rgba(239, 108, 108, 0.1)',
                            tension: 0.3,
                            borderWidth: 1.5,
                            pointRadius: 0, // No markers for many data points
                            fill: false,
                            borderDash: [5, 5]
                        },
                        {
                            label: 'Anger',
                            data: processedData.expressionCounts.map((counts, index) => ({
                                x: processedData.timePoints[index],
                                y: counts.Anger
                            })),
                            borderColor: '#DC2626',
                            backgroundColor: 'rgba(220, 38, 38, 0.1)',
                            tension: 0.3,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        },
                        {
                            label: 'Surprise',
                            data: processedData.expressionCounts.map((counts, index) => ({
                                x: processedData.timePoints[index],
                                y: counts.Surprise
                            })),
                            borderColor: '#FBBF24',
                            backgroundColor: 'rgba(251, 191, 36, 0.1)',
                            tension: 0.3,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        },
                        {
                            label: 'Neutral',
                            data: processedData.expressionCounts.map((counts, index) => ({
                                x: processedData.timePoints[index],
                                y: counts.Neutral
                            })),
                            borderColor: '#9CA3AF',
                            backgroundColor: 'rgba(156, 163, 175, 0.1)',
                            tension: 0.3,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        },
                        {
                            label: 'Disgust',
                            data: processedData.expressionCounts.map((counts, index) => ({
                                x: processedData.timePoints[index],
                                y: counts.Disgust
                            })),
                            borderColor: '#7C2D12',
                            backgroundColor: 'rgba(124, 45, 18, 0.1)',
                            tension: 0.3,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        }
                    ].filter(dataset => dataset.data.some(point => point.y > 0)); // Only show emotions that appear

                    facialExpressionChartInstanceRef.current = new Chart(ctx, {
                        type: 'line',
                        data: {
                            datasets: datasets
                        },
                        options: {
                            ...defaultLineChartOptions('Expression Proportion'),
                            scales: {
                                ...defaultLineChartOptions('Expression Proportion').scales,
                                x: {
                                    ...defaultLineChartOptions('Expression Proportion').scales.x,
                                    type: 'linear',
                                    position: 'bottom',
                                    min: 0,
                                    max: videoDuration,
                                    ticks: {
                                        ...defaultLineChartOptions('Expression Proportion').scales.x.ticks,
                                        stepSize: videoDuration / 3,
                                        callback: function (value) {
                                            return (value as number).toFixed(0) + 's';
                                        }
                                    }
                                },
                                y: {
                                    ...defaultLineChartOptions('Expression Proportion').scales.y,
                                    max: 1.0,
                                    ticks: {
                                        ...defaultLineChartOptions('Expression Proportion').scales.y.ticks,
                                        callback: function (value) {
                                            return (value as number * 100).toFixed(0) + '%';
                                        }
                                    }
                                }
                            },
                            parsing: false
                        }
                    });
                }
            }
        }

        // Heart Rate Chart with CSV data
        if (heartRateChartRef.current && heartRateData) {
            const ctx = heartRateChartRef.current.getContext('2d');
            if (ctx) {
                // Destroy existing chart if it exists
                if (heartRateChartInstanceRef.current) {
                    heartRateChartInstanceRef.current.destroy();
                    heartRateChartInstanceRef.current = null;
                }

                heartRateChartInstanceRef.current = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [{
                            label: 'Heart Rate (BPM)',
                            data: heartRateData.map(point => ({
                                x: point.time_seconds,
                                y: point.heart_rate_bpm
                            })),
                            borderColor: '#EC4899',
                            backgroundColor: 'rgba(236, 72, 153, 0.2)',
                            tension: 0.3,
                            borderWidth: 2,
                            pointRadius: 0, // No markers for many data points
                            fill: true
                        }]
                    },
                    options: {
                        ...defaultLineChartOptions('Heart Rate (BPM)'),
                        scales: {
                            ...defaultLineChartOptions('Heart Rate (BPM)').scales,
                            x: {
                                ...defaultLineChartOptions('Heart Rate (BPM)').scales.x,
                                type: 'linear',
                                position: 'bottom',
                                min: 0,
                                max: videoDuration,
                                ticks: {
                                    ...defaultLineChartOptions('Heart Rate (BPM)').scales.x.ticks,
                                    stepSize: videoDuration / 3,
                                    callback: function (value) {
                                        return (value as number).toFixed(0) + 's';
                                    }
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        parsing: false
                    }
                });
            }
        }

        // Spider charts (empty)
        const spiderChartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                line: {
                    borderWidth: 2
                }
            },
            scales: {
                r: {
                    angleLines: {
                        color: chartGridColor
                    },
                    grid: {
                        color: chartGridColor
                    },
                    pointLabels: {
                        font: {
                            size: 10
                        },
                        color: chartTextColor
                    },
                    ticks: {
                        backdropColor: 'rgba(45, 55, 72, 0.75)',
                        color: chartTextColor,
                        stepSize: 0.2,
                        font: {
                            size: 9
                        }
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        };

        // Verbal Emotional Spectrum Spider Chart
        if (spiderChartVerbalRef.current && verbalEmotionData) {
            const ctx = spiderChartVerbalRef.current.getContext('2d');
            if (ctx) {
                // Destroy existing chart if it exists
                if (spiderChartVerbalInstanceRef.current) {
                    spiderChartVerbalInstanceRef.current.destroy();
                    spiderChartVerbalInstanceRef.current = null;
                }

                // Use all 8 emotions from the verbal data
                const allEmotions = verbalEmotionData.sort((a, b) => b.level - a.level); // Sort by level descending

                const verbalLabels = allEmotions.map(item =>
                    item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)
                );
                const verbalData = allEmotions.map(item => item.level / 9); // Normalize 0-9 to 0-1

                spiderChartVerbalInstanceRef.current = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: verbalLabels,
                        datasets: [{
                            label: 'Verbal Emotions',
                            data: verbalData,
                            fill: true,
                            backgroundColor: accentPrimary.replace('1)', '0.3)'),
                            borderColor: accentPrimary,
                            pointBackgroundColor: accentPrimary,
                            pointBorderColor: '#2D3748',
                            pointHoverBackgroundColor: '#E2E8F0',
                            pointHoverBorderColor: accentPrimary,
                            borderWidth: 2
                        }]
                    },
                    options: spiderChartOptions
                });
            }
        }

        // Non-Verbal Emotional Spectrum Spider Chart
        if (spiderChartNonVerbalRef.current && nonverbalEmotionData) {
            const ctx = spiderChartNonVerbalRef.current.getContext('2d');
            if (ctx) {
                // Destroy existing chart if it exists
                if (spiderChartNonVerbalInstanceRef.current) {
                    spiderChartNonVerbalInstanceRef.current.destroy();
                    spiderChartNonVerbalInstanceRef.current = null;
                }

                // Use all 8 emotions from the non-verbal data
                const allNonverbalEmotions = nonverbalEmotionData.sort((a, b) => b.level - a.level); // Sort by level descending

                const nonverbalLabels = allNonverbalEmotions.map(item =>
                    item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)
                );
                const nonverbalData = allNonverbalEmotions.map(item => item.level / 8); // Normalize 0-8 to 0-1

                spiderChartNonVerbalInstanceRef.current = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: nonverbalLabels,
                        datasets: [{
                            label: 'Non-Verbal Emotions',
                            data: nonverbalData,
                            fill: true,
                            backgroundColor: accentSecondary.replace('1)', '0.3)'),
                            borderColor: accentSecondary,
                            pointBackgroundColor: accentSecondary,
                            pointBorderColor: '#2D3748',
                            pointHoverBackgroundColor: '#E2E8F0',
                            pointHoverBorderColor: accentSecondary,
                            borderWidth: 2
                        }]
                    },
                    options: spiderChartOptions
                });
            }
        }

    }, [videoDuration, facialExpressionData, heartRateData, verbalEmotionData, nonverbalEmotionData]);

    // Cleanup effect when component unmounts
    useEffect(() => {
        return () => {
            const chartRefs = [
                voiceToneChartInstanceRef,
                facialExpressionChartInstanceRef,
                heartRateChartInstanceRef,
                spiderChartVerbalInstanceRef,
                spiderChartNonVerbalInstanceRef
            ];

            chartRefs.forEach(ref => {
                if (ref.current) {
                    ref.current.destroy();
                    ref.current = null;
                }
            });
        };
    }, []);

    // Fallback effect to check video duration after component mounts
    useEffect(() => {
        const checkVideoDuration = () => {
            if (videoRef.current && !isNaN(videoRef.current.duration) && videoRef.current.duration > 0) {
                console.log('Fallback check - video duration:', videoRef.current.duration);
                setVideoDuration(videoRef.current.duration);
            }
        };

        // Check after a delay to ensure video is loaded
        const timer = setTimeout(checkVideoDuration, 1000);
        return () => clearTimeout(timer);
    }, []);

    return (
        <div className="min-h-screen" style={{
            fontFamily: 'Inter, sans-serif',
            backgroundColor: '#1A202C',
            color: '#E2E8F0'
        }}>
            <div className="container max-w-7xl mx-auto p-8">
                <header className="text-center mb-16">
                    <h1 className="text-5xl font-bold text-white">Emotion Insights</h1>
                    <p className="text-xl text-gray-400 mt-2">Understanding the emotional landscape of your video content.</p>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Video Section */}
                    <div className="lg:col-span-3 bg-gray-800 rounded-xl p-8 shadow-xl border border-gray-600">
                        <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                            <span className="material-icons text-blue-400 mr-3">videocam</span>
                            Video Analysis
                        </h2>
                        <div className="w-full aspect-video bg-gray-600 rounded-lg overflow-hidden">
                            <video
                                ref={videoRef}
                                controls
                                className="w-full h-full"
                                style={{ backgroundColor: '#4A5568' }}
                                onLoadedMetadata={handleVideoMetadataLoaded}
                                onCanPlay={handleVideoCanPlay}
                                onLoadedData={handleVideoCanPlay}
                            >
                                <source src="/khabib.mp4" type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    </div>

                    {/* Left Column - Charts */}
                    <div className="lg:col-span-1 bg-gray-800 rounded-xl p-8 shadow-xl border border-gray-600 space-y-6 min-h-[800px]">
                        <div>
                            <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                                <span className="material-icons text-blue-400 mr-3">mic</span>
                                Voice/Tone Progression
                            </h2>
                            <div className="relative h-48 w-full">
                                <canvas ref={voiceToneChartRef}></canvas>
                            </div>
                        </div>

                        <div>
                            <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                                <span className="material-icons text-blue-400 mr-3">face</span>
                                Facial Expressions
                            </h2>
                            <div className="relative h-48 w-full">
                                <canvas ref={facialExpressionChartRef}></canvas>
                            </div>
                        </div>

                        <div>
                            <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                                <span className="material-icons text-blue-400 mr-3">favorite_border</span>
                                Heart Rate (BPM) Progression
                            </h2>
                            <div className="relative h-48 w-full mb-6">
                                <canvas ref={heartRateChartRef}></canvas>
                            </div>
                            <p className="text-xs text-center text-gray-400 mt-1">Heart rate fluctuations over time</p>
                        </div>
                    </div>

                    {/* Middle Column - Emotional Analysis */}
                    <div className="lg:col-span-1 bg-gray-800 rounded-xl p-8 shadow-xl border border-gray-600 space-y-6 min-h-[800px]">
                        <div>
                            <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                                <span className="material-icons text-blue-400 mr-3">psychology</span>
                                Emotional Analysis
                            </h2>
                            <div className="bg-gray-900 border border-gray-600 rounded-lg p-4 text-sm leading-relaxed text-gray-400 h-[calc(100%-3rem)]">
                                Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, quos.
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Spider Charts */}
                    <div className="lg:col-span-1 bg-gray-800 rounded-xl p-8 shadow-xl border border-gray-600 min-h-[800px]">
                        <h2 className="text-xl font-semibold text-white mb-5 flex items-center">
                            <span className="material-icons text-blue-400 mr-3">record_voice_over</span>
                            Verbal Emotional Spectrum
                        </h2>
                        <div className="h-48 w-full mb-4">
                            <canvas ref={spiderChartVerbalRef}></canvas>
                        </div>
                        <p className="text-xs text-center text-gray-400 mt-1">Distribution of emotions based on spoken words</p>

                        <h2 className="text-xl font-semibold text-white mb-4 mt-6 flex items-center">
                            <span className="material-icons text-blue-400 mr-3">accessibility_new</span>
                            Non-Verbal Emotional Spectrum
                        </h2>
                        <div className="h-48 w-full mb-4">
                            <canvas ref={spiderChartNonVerbalRef}></canvas>
                        </div>
                        <p className="text-xs text-center text-gray-400 mt-1">Distribution of emotions based on body language & gestures</p>

                        <div className="mt-6">
                            <Image
                                src="/khabib_dots.png"
                                alt="Emotion visualization dots"
                                className="w-full h-auto rounded-lg border border-gray-600"
                                width={300}
                                height={100}
                            />
                        </div>
                    </div>
                </div>

                <footer className="text-center mt-16 py-8 border-t border-gray-600">
                    <p className="text-gray-400 text-sm">© 2024 Emotion AI. All rights reserved.</p>
                </footer>
            </div>
        </div>
    );
} 
