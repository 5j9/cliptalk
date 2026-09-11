const configContent = document.getElementById('config-content');
const toggleConfig = document.getElementById('toggle-config');
const saveConfig = document.getElementById('save-config');
const configStatus = document.getElementById('config-status');
const configPanel = document.getElementById('config-panel');


toggleConfig.addEventListener('click', async () => {
    const isHidden = configContent.style.display === 'none';

    configContent.style.display = isHidden ? 'block' : 'none';

    if (isHidden) {
        await loadConfig();
    }
});


document.addEventListener('click', (event) => {
    if (
        configContent.style.display !== 'none'
        && !configPanel.contains(event.target)
    ) {
        configContent.style.display = 'none';
    }
});


async function loadConfig() {
    try {
        const response = await fetch('/config');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const config = await response.json();

        document.querySelectorAll('[data-engine-key]').forEach(select => {
            select.value = config.engines[select.dataset.engineKey];
        });

        configStatus.textContent = '';
    } catch (error) {
        console.error('Failed to load configuration:', error);
        configStatus.textContent = 'Failed to load configuration';
    }
}


saveConfig.addEventListener('click', async () => {
    const engines = {};

    document.querySelectorAll('[data-engine-key]').forEach(select => {
        engines[select.dataset.engineKey] = select.value;
    });

    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ engines }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        configStatus.textContent = 'Saved';
    } catch (error) {
        console.error('Failed to save configuration:', error);
        configStatus.textContent = 'Failed to save configuration';
    }
});