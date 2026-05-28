// ==UserScript==
// @name          twitter-tweet-sender.user.js
// @namespace     https://github.com/5j9/cliptalk/user-scripts
// @version       1.3
// @description   Adds a button to send tweet text to local endpoint, positioned right of tweet
// @author        5j9
// @match         https://twitter.com/*
// @match         https://x.com/*
// @grant         GM_xmlhttpRequest
// ==/UserScript==

(function () {
    'use strict';

    // Configuration
    const API_ENDPOINT = 'http://127.0.0.1:3775/q';

    // Function to send text to local endpoint
    function sendToEndpoint(text, button) {
        // Mark button as activated immediately to prevent multiple sends
        button.setAttribute('data-activated', 'true');

        // Change button color to indicate it's been activated
        button.style.background = '#9E9E9E'; // Gray for activated state

        GM_xmlhttpRequest({
            method: 'POST',
            url: API_ENDPOINT,
            headers: {
                'Content-Type': 'text/plain;charset=UTF-8'
            },
            data: text,
            onload: function (response) {
                if (response.status === 200) {
                    console.log('Text sent successfully');
                    showFeedback(button, 'success');
                } else {
                    console.error('Failed to send text. Status:', response.status);
                    showFeedback(button, 'error');
                }
            },
            onerror: function (error) {
                console.error('Request failed:', error);
                showFeedback(button, 'error');
            }
        });
    }

    // Function to show temporary feedback
    function showFeedback(button, type) {
        const originalBg = button.style.background;
        if (type === 'success') {
            button.style.background = '#4CAF50'; // Green for success
        } else if (type === 'error') {
            button.style.background = '#f44336'; // Red for error
        }

        // Reset button color after 500ms (but keep activated color if it was activated)
        setTimeout(() => {
            if (button.getAttribute('data-activated') === 'true') {
                button.style.background = '#9E9E9E'; // Return to activated gray
            } else {
                button.style.background = originalBg;
            }
        }, 500);
    }

    // Function to create send button
    function createSendButton(tweetTextElement) {
        const button = document.createElement('button');
        button.textContent = '📤'; // Send emoji
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 80px;
            padding: 4px 8px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            z-index: 1000;
            background: #f0f0f0;
            transition: background 0.2s;
        `;

        // Initialize as not activated
        button.setAttribute('data-activated', 'false');

        // Add a flag to prevent duplicate sends within a short time
        let sendInProgress = false;

        // Unified function to handle sending (with deduplication)
        const handleSend = (e) => {
            e.stopPropagation();

            // Prevent duplicate sends
            if (sendInProgress) {
                console.log('Send already in progress, ignoring');
                return;
            }

            const tweetText = tweetTextElement.textContent.trim();
            if (tweetText) {
                sendInProgress = true;
                sendToEndpoint(tweetText, button);
                // Reset the flag after a short delay to allow resending if needed
                setTimeout(() => {
                    sendInProgress = false;
                }, 1000);
            }
        };

        // Click event handler
        button.addEventListener('click', handleSend);

        // Mouse enter handler (only auto-send if not activated)
        button.addEventListener('mouseenter', (e) => {
            e.stopPropagation();
            // Only auto-send if not activated yet AND not already sending
            if (button.getAttribute('data-activated') === 'false' && !sendInProgress) {
                const tweetText = tweetTextElement.textContent.trim();
                if (tweetText) {
                    sendInProgress = true;
                    sendToEndpoint(tweetText, button);
                    setTimeout(() => {
                        sendInProgress = false;
                    }, 1000);
                }
            }
        });

        return button;
    }

    // Function to add buttons to tweets
    function addSendButtons() {
        const tweetTextElements = document.querySelectorAll('[data-testid="tweetText"]');
        tweetTextElements.forEach(element => {
            // Check if button already exists to avoid duplicates
            if (!element.closest('article').querySelector('.send-button')) {
                const button = createSendButton(element);
                button.classList.add('send-button');
                // Find the tweet's article container
                const tweetContainer = element.closest('article');
                if (tweetContainer) {
                    tweetContainer.style.position = 'relative';
                    tweetContainer.appendChild(button);
                }
            }
        });
    }

    // Initial run
    addSendButtons();

    // Observe for dynamically loaded tweets (e.g., when scrolling)
    const observer = new MutationObserver(() => {
        addSendButtons();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();