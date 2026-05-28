// ==UserScript==
// @name          Twitter Tweet Text Copier with Button
// @namespace     http://tampermonkey.net/
// @version       1.0
// @description   Adds a button to copy tweet text to clipboard, positioned right of tweet
// @author        5j9
// @match         https://twitter.com/*
// @match         https://x.com/*
// @grant         none
// ==/UserScript==

(function () {
    'use strict';

    // Function to copy text to clipboard
    function copyToClipboard(text, button) {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Text copied to clipboard');
                showFeedback(button);
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
            });
    }

    // Function to show temporary feedback
    function showFeedback(button) {
        button.style.background = 'lightsteelblue';
    }

    // Function to create copy button
    function createCopyButton(tweetTextElement) {
        const button = document.createElement('button');
        button.textContent = '📋'; // Change button title to clipboard emoji
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
        `;

        // Define the event listener function
        const handleMouseEnter = (e) => {
            e.stopPropagation(); // Stop event bubbling
            const tweetText = tweetTextElement.textContent.trim();
            if (tweetText) {
                copyToClipboard(tweetText, button);
                // Remove the listener after the first copy
                button.removeEventListener('mouseenter', handleMouseEnter);
                console.log('Mouseenter listener removed for this button.');
            }
        };

        // Add mouse-enter event for auto-copy
        button.addEventListener('mouseenter', handleMouseEnter);

        // Add click event as fallback for browsers with restrictions (uncomment if needed)
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            const tweetText = tweetTextElement.textContent.trim();
            if (tweetText) {
                copyToClipboard(tweetText, button);
            }
        });

        return button;
    }

    // Function to add buttons to tweets
    function addCopyButtons() {
        const tweetTextElements = document.querySelectorAll('[data-testid="tweetText"]');
        tweetTextElements.forEach(element => {
            // Check if button already exists to avoid duplicates
            if (!element.closest('article').querySelector('.copy-button')) {
                const button = createCopyButton(element);
                button.classList.add('copy-button');
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
    addCopyButtons();

    // Observe for dynamically loaded tweets (e.g., when scrolling)
    const observer = new MutationObserver(() => {
        addCopyButtons();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();