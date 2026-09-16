// @ts-check
const port = '3775';
const home = `http://127.0.0.1:${port}/`;

// Cache all DOM elements
const audio = /** @type{HTMLAudioElement} */ (document.querySelector('audio'));
const statusEl = document.getElementById('status');
const inputQueueEl = document.getElementById('input-queue-size');
const outputQueueEl = document.getElementById('output-queue-size');
const nextButton = /** @type{HTMLButtonElement} */ (document.getElementById('next'));
const toggleButton = /** @type{HTMLElement} */ (document.getElementById('toggle-monitoring'));
const editableField = document.getElementById('editable_field');
const clearButton = /** @type{HTMLElement} */ (document.getElementById('clear'));
const speedSlider = /** @type{HTMLInputElement} */ (document.getElementById('speed-slider'));
const speedDisplay = document.getElementById('speed-display');
const resetSpeedBtn = /** @type{HTMLElement} */ (document.getElementById('reset-speed'));
const helpButton = /** @type{HTMLButtonElement} */ (document.getElementById('help-button'));
const helpModal = /** @type{HTMLElement} */ (document.getElementById('help-modal'));
const closeHelpBtn = /** @type{HTMLElement} */ (document.getElementById('close-help'));


let currentSpeed = 1.0;

// Speed control functions
function loadSpeedFromStorage() {
	try {
		const savedSpeed = localStorage.getItem('cliptalk_speed');
		if (savedSpeed !== null) {
			const speed = parseFloat(savedSpeed);
			if (speed >= 0.5 && speed <= 5.0) {
				currentSpeed = speed;
				return speed;
			}
		}
	} catch (error) {
		console.error('Failed to load speed from localStorage:', error);
	}
	return 1.0;
}

/**
 * @param {number} speed
 */
function saveSpeedToStorage(speed) {
	try {
		localStorage.setItem('cliptalk_speed', speed.toString());
	} catch (error) {
		console.error('Failed to save speed to localStorage:', error);
	}
}

function updateSpeed() {
	currentSpeed = parseFloat(speedSlider.value);
	audio.playbackRate = currentSpeed;
	if (speedDisplay) {
		speedDisplay.textContent = currentSpeed.toFixed(1) + 'x';
	}
	saveSpeedToStorage(currentSpeed);
}

// Initialize speed control with saved value
if (speedSlider) {
	const savedSpeed = loadSpeedFromStorage();
	speedSlider.value = savedSpeed.toString();
	if (speedDisplay) {
		speedDisplay.textContent = savedSpeed.toFixed(1) + 'x';
	}
	audio.playbackRate = savedSpeed;
	currentSpeed = savedSpeed;
	speedSlider.addEventListener('input', updateSpeed);
}

if (resetSpeedBtn) {
	resetSpeedBtn.addEventListener('click', () => {
		if (speedSlider) {
			speedSlider.value = '1.0';
			updateSpeed();
			localStorage.removeItem('cliptalk_speed');
		}
	});
}

/**
 * @param {Event | String | unknown} e
 */
function requestNextStream(e) {
	if ((e instanceof Event) && e.type != 'ended') {
		console.log(e);
	}
	fetch(home + 'next');
}

audio.onended = requestNextStream;
audio.onerror = requestNextStream;

// Favicon
// @ts-ignore
var favicon = document.createElement('link');
favicon.rel = 'icon';
favicon.type = 'image/svg+xml';
favicon.href = `data:image/svg+xml,
	<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
		<text x="50%" y="58%" dominant-baseline="middle" text-anchor="middle"
			font-size="16" fill="black">🗣️</text>
	</svg>`;
document.head.appendChild(favicon);


function jumpBackward() {
	audio.currentTime -= 10;
}

function jumpForward() {
	audio.currentTime += 10;
}

function stop() {
	audio.pause();
	audio.currentTime = 0;
}

function next() {
	audio.pause();
	if (nextButton) nextButton.disabled = true;

	fetch(home + 'next').catch(error => {
		console.error('Next request failed:', error);
		if (nextButton) nextButton.disabled = false;
	});
}

async function play() {
	audio.src = 'audio?' + Date.now();
	audio.load();

	audio.addEventListener('loadedmetadata', function onLoaded() {
		audio.removeEventListener('loadedmetadata', onLoaded);
		audio.playbackRate = currentSpeed;
	}, { once: true });

	try {
		await audio.play();
	} catch (e) {
		console.error('Playback failed:', e);
	}
}


var monitoring = false;

async function toggleMonitoring() {
	monitoring = !monitoring;

	try {
		const r = await fetch(home + 'monitoring', {
			method: 'PUT',
			body: JSON.stringify(monitoring)
		});

		if (!r.ok) {
			console.error('Failed to toggle monitoring:', r.status);
			monitoring = !monitoring;
		}

		if (toggleButton) {
			toggleButton.textContent = monitoring ? '⭘' : '⏽';
		}
	} catch (error) {
		console.error('Network error:', error);
		monitoring = !monitoring;
	}
}
if (toggleButton) {
	toggleButton.onclick = toggleMonitoring;
}

// WebSocket
var /** @type{WebSocket | undefined} */ ws;
let reconnectAttempts = 0;
let reconnectTimer = /** @type{ReturnType<typeof setTimeout> | undefined} */ (undefined);

const MAX_RECONNECT_ATTEMPTS = 10;


/**
 * Schedule a WebSocket reconnect.
 */
function scheduleReconnect() {
	if (reconnectTimer !== undefined) {
		return;
	}

	reconnectAttempts++;

	if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
		console.error('Max reconnection attempts reached');
		if (statusEl) statusEl.textContent = '⛔';
		return;
	}

	const delay = Math.min(
		2000 * Math.pow(1.5, reconnectAttempts - 1),
		30000
	);

	if (statusEl) statusEl.textContent = '🔴';

	reconnectTimer = setTimeout(() => {
		reconnectTimer = undefined;
		startWs();
	}, delay);
}


/**
 * Handle WebSocket closure.
 *
 * @param {CloseEvent} e
 */
function onClose(e) {
	console.log('WebSocket closed:', e);

	// Only act on the current socket.
	if (ws !== e.target) {
		return;
	}

	ws = undefined;

	if (statusEl) statusEl.textContent = '🔴';

	scheduleReconnect();
}


/**
 * Start the WebSocket connection if one isn't already active.
 */
function startWs() {
	if (ws && (
		ws.readyState === WebSocket.OPEN ||
		ws.readyState === WebSocket.CONNECTING
	)) {
		return;
	}

	console.log('new websocket');

	try {
		const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
		ws = socket;

		socket.onerror = (e) => {
			console.log('WebSocket error:', e);
		};

		socket.onclose = onClose;

		socket.onopen = () => {
			// Ignore an old socket that happened to open after a newer
			// connection was created.
			if (ws !== socket) {
				socket.close();
				return;
			}

			reconnectAttempts = 0;

			if (reconnectTimer !== undefined) {
				clearTimeout(reconnectTimer);
				reconnectTimer = undefined;
			}

			if (statusEl) statusEl.textContent = '🟢';

			fetch(home + 'monitoring', {
				method: 'PUT',
				body: JSON.stringify(monitoring)
			}).catch(error => {
				console.error('Failed to sync monitoring state:', error);
			});
		};

		socket.onmessage = (e) => {
			// Ignore messages from an old socket.
			if (ws !== socket) {
				return;
			}

			try {
				var j = JSON.parse(e.data);

				switch (j.action) {
					case 'toggle-monitoring':
						monitoring = j.state;
						if (toggleButton) {
							toggleButton.textContent = monitoring ? '⭘' : '⏽';
						}
						break;

					case 'new-text':
						var text = j.text;

						if (editableField) {
							editableField.dir = j.is_fa ? 'rtl' : 'ltr';
							editableField.textContent = text;
						}

						if (nextButton) {
							nextButton.disabled = false;
						}

						play();
						break;

					case 'input-queue-size':
						if (inputQueueEl) {
							inputQueueEl.textContent = j.value;
						}
						break;

					case 'output-queue-size':
						if (outputQueueEl) {
							outputQueueEl.textContent = j.value;
						}
						break;
				}
			} catch (error) {
				console.error('Failed to parse WebSocket message:', error);
			}
		};

	} catch (error) {
		console.error('Failed to create WebSocket:', error);
		ws = undefined;
		scheduleReconnect();
	}
}


/*
 * BFCache
 *
 * When Chrome puts the page into the Back-Forward Cache, the WebSocket
 * is closed. When the page is restored, establish a new connection.
 */
window.addEventListener('pageshow', (e) => {
	if (e.persisted) {
		console.log('Page restored from BFCache');

		reconnectAttempts = 0;

		if (reconnectTimer !== undefined) {
			clearTimeout(reconnectTimer);
			reconnectTimer = undefined;
		}

		// The cached WebSocket is no longer usable.
		if (ws && ws.readyState !== WebSocket.OPEN) {
			ws = undefined;
		}

		startWs();
	}
});


if (clearButton) {
	clearButton.addEventListener('click', () => {
		if (editableField) {
			editableField.textContent = '';
		}
	});
}


document.addEventListener('keydown', (e) => {
	// Handle Escape for help modal
	if (
		e.key === 'Escape' &&
		helpModal &&
		helpModal.classList.contains('active')
	) {
		helpModal.classList.remove('active');
		return;
	}

	// Handle keyboard shortcuts
	if (e.target === editableField) {
		return;
	}

	switch (e.key) {
		case 'ArrowLeft':
			e.preventDefault();
			jumpBackward();
			break;

		case 'ArrowRight':
			e.preventDefault();
			jumpForward();
			break;

		case 'n':
		case 'N':
			next();
			break;

		case 's':
		case 'S':
			stop();
			break;

		case ' ':
			e.preventDefault();

			if (audio.paused) {
				audio.play().catch(err => console.error('Play failed:', err));
			} else {
				audio.pause();
			}
			break;
	}
});


helpButton.addEventListener('click', () => {
	helpModal.classList.add('active');
});

closeHelpBtn.addEventListener('click', () => {
	helpModal.classList.remove('active');
});

helpModal.addEventListener('click', (e) => {
	if (e.target === helpModal) {
		helpModal.classList.remove('active');
	}
});


// Handle audio errors
audio.addEventListener('error', (e) => {
	console.error('Audio error:', e);

	if (statusEl) statusEl.textContent = '❌';

	setTimeout(() => {
		if (audio.src) {
			audio.load();
			audio.play().catch(() => { });
		}
	}, 3000);
});


// Start WebSocket connection
startWs();
