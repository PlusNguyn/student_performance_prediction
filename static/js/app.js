(function () {
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("student-ui-theme");
  const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = storedTheme || (preferredDark ? "dark" : "light");

  const syncThemeIcon = () => {
    document.querySelectorAll("[data-theme-toggle] i").forEach((icon) => {
      icon.className = root.dataset.theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    });
  };
  syncThemeIcon();

  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      localStorage.setItem("student-ui-theme", root.dataset.theme);
      syncThemeIcon();
    });
  });

  const closeSidebar = () => document.body.classList.remove("sidebar-open");
  document.querySelectorAll("[data-sidebar-toggle]").forEach((button) => {
    button.addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
  });
  document.querySelectorAll("[data-sidebar-close], .sidebar-nav a").forEach((item) => {
    item.addEventListener("click", closeSidebar);
  });

  if (window.bootstrap) {
    document.querySelectorAll("[title]").forEach((item) => {
      new bootstrap.Tooltip(item, { trigger: "hover" });
    });
  }

  if (!window.Chart) {
    return;
  }

  Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  Chart.defaults.color = getComputedStyle(root).getPropertyValue("--muted").trim();
  Chart.defaults.plugins.tooltip.backgroundColor = "#101827";
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;

  document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
    try {
      const config = JSON.parse(canvas.getAttribute("data-chart"));
      config.options = config.options || {};
      config.options.maintainAspectRatio = false;
      config.options.responsive = true;
      new Chart(canvas, config);
    } catch (error) {
      canvas.closest(".chart-box")?.classList.add("chart-error");
      console.error("Invalid chart config", error);
    }
  });
})();
