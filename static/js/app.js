"use strict";

(function () {
  var els = {
    syncForm: document.getElementById("sync-form"),
    locationFilter: document.getElementById("location-filter"),
    cityGrid: document.getElementById("city-grid"),
    defaultLocationsList: document.getElementById("default-locations-list"),
    customLat: document.getElementById("custom-lat"),
    customLon: document.getElementById("custom-lon"),
    customAdd: document.getElementById("custom-add"),
    customError: document.getElementById("custom-error"),
    syncSubmit: document.getElementById("sync-submit"),
    syncSpinner: document.getElementById("sync-spinner"),
    syncSubmitLabel: document.getElementById("sync-submit-label"),
    syncStatus: document.getElementById("sync-status"),

    searchForm: document.getElementById("search-form"),
    searchQuery: document.getElementById("search-query"),
    queryField: document.getElementById("query-field"),
    queryError: document.getElementById("query-error"),
    searchTopK: document.getElementById("search-topk"),
    searchSourceType: document.getElementById("search-source-type"),
    searchSummarize: document.getElementById("search-summarize"),
    searchSubmit: document.getElementById("search-submit"),
    searchSpinner: document.getElementById("search-spinner"),
    searchSubmitLabel: document.getElementById("search-submit-label"),

    resultsCount: document.getElementById("results-count"),
    resultsBanner: document.getElementById("results-banner"),
    summaryCallout: document.getElementById("summary-callout"),
    summaryText: document.getElementById("summary-text"),
    summaryErrorNote: document.getElementById("summary-error-note"),
    emptyState: document.getElementById("empty-state"),
    emptyTitle: document.getElementById("empty-title"),
    emptyText: document.getElementById("empty-text"),
    resultsCards: document.getElementById("results-cards"),
  };

  var BADGE_CLASS_BY_SOURCE_TYPE = {
    alert: "badge--alert",
    forecast: "badge--manual",
    discussion: "badge--seed",
  };

  // ----------------------------------------------------- location picker --

  var locationState = {
    known: [],           // every city the server will recognize by name
    defaults: [],         // what /weather/sync uses when nothing is selected
    selected: [],          // known cities the user has clicked on
    custom: [],            // "lat,lon" strings added via the Advanced panel
  };

  init();

  function init() {
    els.syncForm.addEventListener("submit", onSyncSubmit);
    els.locationFilter.addEventListener("input", function () {
      renderCityGrid();
    });
    els.customAdd.addEventListener("click", onAddCustomCoordinate);
    els.searchForm.addEventListener("submit", onSearchSubmit);
    els.searchQuery.addEventListener("input", onQueryInput);
    updateSearchButtonEnabled();
    loadLocations();
  }

  async function loadLocations() {
    try {
      var res = await fetch("/api/locations");
      var data = await safeJson(res);
      if (!res.ok || !data) {
        throw new Error(errorMessageFrom(data, res));
      }
      locationState.known = Array.isArray(data.locations) ? data.locations : [];
      locationState.defaults = Array.isArray(data.default_locations) ? data.default_locations : [];
    } catch (err) {
      // The city grid is a convenience over free text, not a hard
      // dependency -- if the list can't be fetched, fall back to an empty
      // grid (the server still accepts a blank sync, using its own default)
      // rather than blocking the form on a failed GET.
      locationState.known = [];
      locationState.defaults = [];
    }
    els.defaultLocationsList.textContent = locationState.defaults.length
      ? locationState.defaults.join(", ")
      : "a preset few";
    renderCityGrid();
  }

  function renderCityGrid() {
    var filter = els.locationFilter.value.trim().toLowerCase();
    var matches = locationState.known.filter(function (city) {
      return !filter || city.toLowerCase().indexOf(filter) !== -1;
    });

    var buttons = matches.map(function (city) {
      return buildCityButton(city, locationState.selected.indexOf(city) !== -1, false);
    });

    locationState.custom.forEach(function (coord) {
      buttons.push(buildCityButton(coord, true, true));
    });

    if (!buttons.length) {
      var empty = document.createElement("span");
      empty.className = "citygrid__empty";
      empty.textContent = locationState.known.length
        ? "No city matches “" + els.locationFilter.value.trim() + "”."
        : "Could not load the location list -- leave everything unselected to sync the defaults.";
      els.cityGrid.replaceChildren(empty);
      return;
    }

    els.cityGrid.replaceChildren(...buttons);
  }

  function buildCityButton(label, isSelected, isCustom) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "city-btn" + (isSelected ? " is-selected" : "") + (isCustom ? " is-custom" : "");
    button.setAttribute("aria-pressed", String(isSelected));

    if (isSelected) {
      var check = document.createElement("span");
      check.className = "city-btn__check";
      check.textContent = "✓";
      button.appendChild(check);
    }
    button.appendChild(document.createTextNode(label));

    button.addEventListener("click", function () {
      if (isCustom) {
        locationState.custom = locationState.custom.filter(function (c) { return c !== label; });
      } else {
        toggleSelected(label);
      }
      renderCityGrid();
    });
    return button;
  }

  function toggleSelected(city) {
    var index = locationState.selected.indexOf(city);
    if (index === -1) {
      locationState.selected.push(city);
    } else {
      locationState.selected.splice(index, 1);
    }
  }

  function onAddCustomCoordinate() {
    els.customError.hidden = true;
    var lat = parseFloat(els.customLat.value);
    var lon = parseFloat(els.customLon.value);

    if (!els.customLat.value.trim() || !els.customLon.value.trim() || isNaN(lat) || isNaN(lon)) {
      els.customError.textContent = "Enter both a latitude and a longitude as numbers.";
      els.customError.hidden = false;
      return;
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      els.customError.textContent = "Latitude must be -90 to 90, longitude -180 to 180.";
      els.customError.hidden = false;
      return;
    }

    var coord = lat + "," + lon;
    if (locationState.custom.indexOf(coord) === -1) {
      locationState.custom.push(coord);
    }
    els.customLat.value = "";
    els.customLon.value = "";
    renderCityGrid();
  }

  function selectedLocations() {
    return locationState.selected.concat(locationState.custom);
  }

  // ------------------------------------------------------------ helpers ----

  async function safeJson(res) {
    try {
      return await res.json();
    } catch (err) {
      return null;
    }
  }

  function errorMessageFrom(data, res) {
    if (data && data.error && data.error.message) {
      return data.error.message;
    }
    return "Request failed with status " + res.status + ".";
  }

  function describeError(err) {
    return err && err.message ? err.message : String(err);
  }

  function setButtonLoading(button, spinner, label, isLoading, idleText, loadingText) {
    button.disabled = isLoading;
    spinner.hidden = !isLoading;
    label.textContent = isLoading ? loadingText : idleText;
  }

  function clearElement(el) {
    el.replaceChildren();
  }

  // ------------------------------------------------------------------ sync --

  async function onSyncSubmit(event) {
    event.preventDefault();

    var locations = selectedLocations();

    setButtonLoading(els.syncSubmit, els.syncSpinner, els.syncSubmitLabel, true, "Sync weather data", "Syncing…");
    hideSyncStatus();

    try {
      var res = await fetch("/weather/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locations: locations }),
      });
      var data = await safeJson(res);
      if (!res.ok) {
        renderSyncStatus(null, errorMessageFrom(data, res));
      } else {
        renderSyncStatus(data, null);
      }
    } catch (err) {
      renderSyncStatus(null, "Could not reach the server: " + describeError(err));
    } finally {
      setButtonLoading(els.syncSubmit, els.syncSpinner, els.syncSubmitLabel, false, "Sync weather data", "Syncing…");
    }
  }

  function hideSyncStatus() {
    els.syncStatus.hidden = true;
    clearElement(els.syncStatus);
  }

  function renderSyncStatus(data, topLevelError) {
    clearElement(els.syncStatus);
    els.syncStatus.hidden = false;

    if (topLevelError) {
      els.syncStatus.className = "field__error";
      var p = document.createElement("p");
      p.textContent = topLevelError;
      els.syncStatus.appendChild(p);
      return;
    }

    els.syncStatus.className = "field__hint";

    var summary = document.createElement("p");
    var synced = typeof data.synced === "number" ? data.synced : 0;
    var considered = typeof data.considered === "number" ? data.considered : 0;
    summary.textContent = "Synced " + synced + " of " + considered + " considered document(s).";
    els.syncStatus.appendChild(summary);

    var errorLines = [];
    var byLocation = data.by_location || {};
    Object.keys(byLocation).forEach(function (location) {
      var entry = byLocation[location] || {};
      if (entry.error) {
        errorLines.push(location + ": " + entry.error);
      }
      var errors = entry.errors || {};
      Object.keys(errors).forEach(function (sourceType) {
        errorLines.push(location + " (" + sourceType + "): " + errors[sourceType]);
      });
    });
    if (data.dropped_source_types && data.dropped_source_types.length) {
      errorLines.push("Ignored unknown source type(s): " + data.dropped_source_types.join(", "));
    }

    if (errorLines.length) {
      var list = document.createElement("ul");
      list.className = "sync-errors";
      errorLines.forEach(function (line) {
        var li = document.createElement("li");
        li.textContent = line;
        list.appendChild(li);
      });
      els.syncStatus.appendChild(list);
    }
  }

  // ---------------------------------------------------------------- search --

  function onQueryInput() {
    clearQueryError();
    updateSearchButtonEnabled();
  }

  function updateSearchButtonEnabled() {
    els.searchSubmit.disabled = els.searchQuery.value.trim().length === 0;
  }

  function setQueryError(message) {
    els.queryField.classList.add("has-error");
    els.queryError.textContent = message;
    els.queryError.hidden = false;
  }

  function clearQueryError() {
    els.queryField.classList.remove("has-error");
    els.queryError.hidden = true;
  }

  function buildSearchQueryString(query) {
    var topK = els.searchTopK.value;
    var sourceType = els.searchSourceType.value;
    var summarize = els.searchSummarize.checked ? "true" : "false";
    return (
      "query=" + encodeURIComponent(query) +
      "&top_k=" + encodeURIComponent(topK) +
      "&source_type=" + encodeURIComponent(sourceType) +
      "&summarize=" + encodeURIComponent(summarize)
    );
  }

  async function onSearchSubmit(event) {
    event.preventDefault();

    var query = els.searchQuery.value.trim();
    if (!query) {
      setQueryError("Enter a search query.");
      return;
    }
    clearQueryError();

    setButtonLoading(els.searchSubmit, els.searchSpinner, els.searchSubmitLabel, true, "Search", "Searching…");
    hideBanner();

    try {
      var qs = buildSearchQueryString(query);
      var res = await fetch("/weather/search?" + qs, { method: "GET" });
      var data = await safeJson(res);
      if (!res.ok) {
        showBanner("error", "Search failed", errorMessageFrom(data, res));
        resetResultsArea();
      } else {
        renderResults(data);
      }
    } catch (err) {
      showBanner("error", "Search failed", "Could not reach the server: " + describeError(err));
      resetResultsArea();
    } finally {
      setButtonLoading(els.searchSubmit, els.searchSpinner, els.searchSubmitLabel, false, "Search", "Searching…");
    }
  }

  // ------------------------------------------------------ results banners --

  function buildBanner(kind, title, text) {
    var banner = document.createElement("div");
    banner.className = kind === "info" ? "banner banner--info" : "banner";

    var bar = document.createElement("div");
    bar.className = "banner__bar";
    banner.appendChild(bar);

    var body = document.createElement("div");
    body.className = "banner__body";

    var titleEl = document.createElement("span");
    titleEl.className = "banner__title";
    titleEl.textContent = title;
    body.appendChild(titleEl);

    var textEl = document.createElement("span");
    textEl.className = "banner__text";
    textEl.textContent = text;
    body.appendChild(textEl);

    banner.appendChild(body);
    return banner;
  }

  function showBanner(kind, title, text) {
    clearElement(els.resultsBanner);
    els.resultsBanner.appendChild(buildBanner(kind, title, text));
    els.resultsBanner.hidden = false;
  }

  function hideBanner() {
    els.resultsBanner.hidden = true;
    clearElement(els.resultsBanner);
  }

  // --------------------------------------------------------- results area --

  function setEmptyState(title, text) {
    els.emptyTitle.textContent = title;
    els.emptyText.textContent = text;
    els.emptyState.hidden = false;
  }

  function resetResultsArea() {
    els.resultsCount.textContent = "";
    els.summaryCallout.hidden = true;
    els.summaryErrorNote.hidden = true;
    clearElement(els.resultsCards);
    els.emptyState.hidden = true;
  }

  function renderResults(data) {
    resetResultsArea();
    hideBanner();

    if (data.reason) {
      els.resultsCount.textContent = "";
      showBanner("info", "Nothing to search yet", data.reason);
      return;
    }

    var results = Array.isArray(data.results) ? data.results : [];
    var count = typeof data.count === "number" ? data.count : results.length;
    els.resultsCount.textContent = count + (count === 1 ? " result" : " results");

    if (data.summary) {
      els.summaryText.textContent = data.summary;
      els.summaryCallout.hidden = false;
    } else if (data.summary_error) {
      els.summaryErrorNote.textContent = "AI summary unavailable: " + data.summary_error;
      els.summaryErrorNote.hidden = false;
    }

    if (results.length === 0) {
      setEmptyState(
        "No matches",
        "No documents matched that query. Try different wording, a broader source type filter, or sync more data first."
      );
      return;
    }

    els.resultsCards.replaceChildren(...results.map(buildResultCard));
  }

  function buildSimilarityBar(similarity) {
    var value = typeof similarity === "number" && !isNaN(similarity) ? similarity : 0;
    var pctRaw = value * 100;
    var pctClamped = Math.max(0, Math.min(100, pctRaw));

    var wrap = document.createElement("div");
    wrap.className = "simbar";

    var track = document.createElement("div");
    track.className = "simbar__track";
    var fill = document.createElement("div");
    fill.className = "simbar__fill";
    fill.style.width = pctClamped + "%";
    track.appendChild(fill);
    wrap.appendChild(track);

    var pctLabel = document.createElement("span");
    pctLabel.className = "simbar__pct";
    pctLabel.textContent = pctRaw.toFixed(1) + "%";
    wrap.appendChild(pctLabel);

    return wrap;
  }

  function buildResultCard(row) {
    var card = document.createElement("div");
    card.className = "card";

    var head = document.createElement("div");
    head.className = "card__head";

    var badge = document.createElement("span");
    var sourceType = row.source_type || "";
    badge.className = "badge " + (BADGE_CLASS_BY_SOURCE_TYPE[sourceType] || "badge--stale");
    badge.textContent = (sourceType || "unknown").toUpperCase();
    head.appendChild(badge);

    var meta = document.createElement("span");
    meta.className = "card__meta";
    meta.textContent = row.location || "";
    head.appendChild(meta);

    card.appendChild(head);

    var title = document.createElement("div");
    title.className = "result__title";
    title.textContent = row.headline || row.event || "Weather update";
    card.appendChild(title);

    card.appendChild(buildSimilarityBar(row.similarity));

    var text = document.createElement("p");
    text.className = "result__text";
    text.textContent = row.chunk_text || "";
    card.appendChild(text);

    return card;
  }
})();
