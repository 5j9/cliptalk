// ==UserScript==
// @name          Twitter Tweet Text Sender with Button
// @namespace     http://tampermonkey.net/
// @version       1.1
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

        // Reset button color after 500ms
        setTimeout(() => {
            button.style.background = originalBg;
        }, 500);
    }

    // Function to create send button
    function createSendButton(tweetTextElement) {
        const button = document.createElement('button');
        button.textContent = '📤'; // Changed to send emoji
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

        // Define the event listener function
        const handleMouseEnter = (e) => {
            e.stopPropagation(); // Stop event bubbling
            const tweetText = tweetTextElement.textContent.trim();
            if (tweetText) {
                sendToEndpoint(tweetText, button);
                // Remove the listener after the first send
                button.removeEventListener('mouseenter', handleMouseEnter);
                console.log('Mouseenter listener removed for this button.');
            }
        };

        // Add mouse-enter event for auto-send
        button.addEventListener('mouseenter', handleMouseEnter);

        // Add click event as fallback
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            const tweetText = tweetTextElement.textContent.trim();
            if (tweetText) {
                sendToEndpoint(tweetText, button);
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