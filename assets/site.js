(() => {
  const button = document.querySelector('[data-theme-toggle]');
  const update = () => {
    const light = document.documentElement.dataset.theme === 'light';
    button.setAttribute('aria-label', `Switch to ${light ? 'dark' : 'light'} mode`);
    button.querySelector('span').textContent = light ? 'Dark mode' : 'Light mode';
    const color = document.querySelector('meta[name="theme-color"]');
    if (color) color.content = light ? '#faf9f6' : '#111c29';
  };
  if (button) {
    button.hidden = false;
    update();
    button.addEventListener('click', () => {
      const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = theme;
      try { localStorage.setItem('lc-theme', theme); } catch (_) {}
      update();
    });
  }
})();
