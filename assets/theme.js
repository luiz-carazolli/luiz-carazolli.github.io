(() => {
  let theme = 'dark';
  try {
    const saved = localStorage.getItem('lc-theme');
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch (_) { /* Browsing with storage disabled still supports switching. */ }
  document.documentElement.dataset.theme = theme;
})();
