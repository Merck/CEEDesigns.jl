using DataFrames

"""
    ensemble_to_dataframe(runs)

Flatten a vector of Pareto fronts (one per ensemble run) into
`df::DataFrame`: DataFrame with columns Threshold, Action_Set,
Frequency, Average_Utility
"""
function ensemble_to_dataframe(runs)
    counts = Dict{Tuple{Float64,String},Tuple{Int,Float64}}()
    for front in runs
        for design in front
            threshold = design[1][2]
            actions = if haskey(design[2], :arrangement)
                join(sort(design[2][:arrangement]), ",")
            else
                ""
            end
            utility = design[1][1]
            key = (threshold, actions)
            prev = get(counts, key, (0, 0.0))
            counts[key] = (prev[1] + 1, prev[2] + utility)
        end
    end
    rows = [
        (
            Threshold = k[1],
            Action_Set = k[2],
            Average_Utility = v[2] / v[1],
            Frequency = v[1],
        ) for (k, v) in counts
    ]
    return DataFrame(rows)
end

"""
    plot_ensemble_pareto(df::DataFrame, tau::Float64; show_annotations::Bool=true, normalization::Symbol=:global)

Create Pareto front visualization for ensemble results with viridis color scheme and horizontal dispersion.

# Arguments

  - `df::DataFrame`: DataFrame with columns Threshold, Action_Set, Frequency, Average_Utility

  - `tau::Float64`: Belief threshold value for labeling
  - `show_annotations::Bool`: Whether to show action set annotations (default: true)
  - `normalization::Symbol`: Normalization method for frequencies (default: :global)

      + `:global` - Normalize by max frequency across all points (recommended)
      + `:per_threshold` - Normalize within each threshold (can be misleading)
      + `:none` - Use raw frequency counts

# Returns

  - Scatter plot showing Pareto front with:

      + Points with uniform size, colored by probability using viridis scheme
      + Horizontal dispersion to show distinct action sets per threshold
      + Count annotations showing number of action sets per threshold
      + MLASP (Maximum Likelihood Action Sets Path) in red
      + Optional annotations for action set changes

# Notes

  - Uses viridis color scheme for better perceptual uniformity and accessibility
  - Horizontal spread indicates number of distinct action sets at each threshold  # :global, :per_threshold, or :none
  - Global normalization is recommended as it preserves relative frequencies across thresholds    # Calculate normalized frequencies for coloring
"""
function plot_ensemble_pareto(
    df::DataFrame,
    tau::Float64;
    show_annotations::Bool = true,
    normalization::Symbol = :global,
)  # :global, :per_threshold, or :none
    # Calculate normalized frequencies for coloring
    df_copy = copy(df)

    # Handle missing action sets
    if "Action_Set" in names(df_copy)
        df_copy.Action_Set = coalesce.(df_copy.Action_Set, "[\"No Action\"]")
    end

    # Calculate horizontal dispersion for each threshold to show distinct action sets
    threshold_groups = groupby(df_copy, :Threshold)
    dispersion_offsets = Dict{Float64,Vector{Float64}}()
    action_counts = Dict{Float64,Int}()

    x_range = maximum(df_copy.Average_Utility) - minimum(df_copy.Average_Utility)

    for group in threshold_groups
        threshold = group.Threshold[1]
        n_actions = nrow(group)
        action_counts[threshold] = n_actions

        # Calculate systematic offsets to spread points horizontally
        if n_actions > 1
            # Create evenly spaced offsets
            max_offset = min(0.02 * x_range, x_range / 50)  # 2% of x-range or x_range/50, whichever is smaller
            offsets = collect(range(-max_offset, max_offset; length = n_actions))

            # Sort group by frequency to place higher frequency items in center
            sorted_indices = sortperm(group.Frequency; rev = true)
            ordered_offsets = similar(offsets)

            # Arrange offsets so highest frequency is in center
            center_first_order = Int[]
            left = 1
            right = n_actions
            for i = 1:n_actions
                if i % 2 == 1
                    push!(center_first_order, div(n_actions + i, 2))
                else
                    push!(center_first_order, div(n_actions - i + 2, 2))
                end
            end

            for (idx, sorted_idx) in enumerate(sorted_indices)
                ordered_offsets[sorted_idx] = offsets[center_first_order[idx]]
            end

            dispersion_offsets[threshold] = ordered_offsets
        else
            dispersion_offsets[threshold] = [0.0]
        end
    end

    # Apply dispersion to x-coordinates
    # IMPORTANT: Process in the same order to maintain alignment with norm_frequencies
    x_coords = Float64[]
    threshold_indices = Dict{Float64,Int}()

    for (idx, row) in enumerate(eachrow(df_copy))
        threshold = row.Threshold

        # Track which index this is within its threshold group
        if !haskey(threshold_indices, threshold)
            threshold_indices[threshold] = 1
        else
            threshold_indices[threshold] += 1
        end

        # Get the appropriate offset for this point
        group_data = filter(r -> r.Threshold == threshold, df_copy)
        point_index = findfirst(
            r ->
                r.Action_Set == row.Action_Set &&
                r.Average_Utility == row.Average_Utility &&
                r.Frequency == row.Frequency,
            eachrow(group_data),
        )

        if !isnothing(point_index) && haskey(dispersion_offsets, threshold)
            offset = dispersion_offsets[threshold][point_index]
            push!(x_coords, row.Average_Utility + offset)
        else
            push!(x_coords, row.Average_Utility)
        end
    end

    # Calculate normalized frequencies based on chosen method
    norm_frequencies = Float64[]
    colorbar_label = "Probability"

    # Check actual successful runs at each threshold
    threshold_counts = Dict{Float64,Int}()
    for threshold in unique(df_copy.Threshold)
        threshold_data = filter(row -> row.Threshold == threshold, df_copy)
        threshold_counts[threshold] = round(Int, sum(threshold_data.Frequency))
    end

    # Use maximum count as the intended ensemble size
    max_ensemble_count = maximum(values(threshold_counts))

    # Report if frequencies vary across thresholds (indicating failed runs)
    if length(unique(values(threshold_counts))) > 1
        println(
            "    ⚠️ Variable success rates across thresholds: $(minimum(values(threshold_counts)))-$(maximum(values(threshold_counts))) successful runs",
        )
    end

    if normalization == :global
        # Use actual counts per threshold for proper normalization
        # This ensures each threshold's frequencies sum to 1 (true probability)
        # IMPORTANT: Process in the same order as df_copy to maintain alignment
        norm_frequencies = Float64[]
        for row in eachrow(df_copy)
            threshold_sum = threshold_counts[row.Threshold]
            if threshold_sum > 0
                push!(norm_frequencies, row.Frequency / threshold_sum)
            else
                push!(norm_frequencies, row.Frequency)
            end
        end
        colorbar_label = "Probability"
    elseif normalization == :per_threshold
        # Per-threshold normalization (can be misleading)
        for threshold in unique(df_copy.Threshold)
            threshold_data = filter(row -> row.Threshold == threshold, df_copy)
            sum_freq = sum(threshold_data.Frequency)
            if sum_freq > 0
                for freq in threshold_data.Frequency
                    push!(norm_frequencies, freq / sum_freq)
                end
            else
                append!(norm_frequencies, threshold_data.Frequency)
            end
        end
        colorbar_label = "Normalized\nFrequency"
    else  # :none
        # Raw frequencies without normalization
        norm_frequencies = df_copy.Frequency
        colorbar_label = "Frequency"
    end

    # Calculate normalized marker sizes based on within-threshold frequency
    # Size encodes relative frequency within each threshold
    # Normalize so that the MAX frequency at each threshold gets the same size

    # First, find max frequency at each threshold
    max_freq_per_threshold = Dict{Float64,Float64}()
    for threshold in unique(df_copy.Threshold)
        threshold_data = filter(row -> row.Threshold == threshold, df_copy)
        if !isempty(threshold_data)
            max_freq_per_threshold[threshold] = maximum(threshold_data.Frequency)
        else
            max_freq_per_threshold[threshold] = 1.0
        end
    end

    marker_sizes = Float64[]
    for row in eachrow(df_copy)
        max_freq = max_freq_per_threshold[row.Threshold]
        if max_freq > 0
            # Normalize by max frequency at this threshold
            normalized_freq = row.Frequency / max_freq
            # Scale so max at each threshold gets size 12, min gets size 4
            size = 4 + 8 * normalized_freq
            push!(marker_sizes, size)
        else
            push!(marker_sizes, 2)
        end
    end

    # Determine title - keep it concise
    title_text = if "Action_Set" in names(df_copy)
        unique_action_sets = unique(df_copy.Action_Set)
        n_unique = length(unique_action_sets)
        "Ensemble Pareto Front (τ = $tau, N=$max_ensemble_count)"
    else
        "Ensemble Pareto Front (τ = $tau, N=$max_ensemble_count)"
    end

    # Create initial plot with scatter points
    # Use normal frequencies but reverse the colormap for darker = higher probability

    # Calculate scale factor for x-axis to ensure single-digit tick labels
    x_max_val = maximum(abs.(x_coords))
    x_min_val = minimum(x_coords)

    # Determine scale factor to get single digits (aim for max value < 10)
    # Use order of magnitude to determine the best scale
    if x_max_val > 0
        order_of_magnitude = floor(log10(x_max_val))

        # Scale to get values in range 0-10 for single digit ticks
        if order_of_magnitude >= 1
            scale_factor = 10^order_of_magnitude
            scale_label = order_of_magnitude == 1 ? "×10" : "×10^$(Int(order_of_magnitude))"

            # Use superscript notation for better display
            if order_of_magnitude == 2
                scale_label = "×10²"
            elseif order_of_magnitude == 3
                scale_label = "×10³"
            elseif order_of_magnitude == 4
                scale_label = "×10⁴"
            elseif order_of_magnitude == 5
                scale_label = "×10⁵"
            elseif order_of_magnitude == 6
                scale_label = "×10⁶"
            end
        else
            scale_factor = 1
            scale_label = ""
        end
    else
        scale_factor = 1
        scale_label = ""
    end

    # Scale x-coordinates for plotting
    x_coords_scaled = x_coords ./ scale_factor

    # Create custom x-axis label with scale factor
    x_label_text = scale_label == "" ? "Cost" : "Cost ($scale_label)"

    # Format x-axis ticks for single-digit display
    x_min_scaled = minimum(x_coords_scaled)
    x_max_scaled = maximum(x_coords_scaled)

    # For single-digit ticks, use simple intervals
    x_range = x_max_scaled - x_min_scaled

    # Determine tick interval to get roughly 5-8 ticks with single digits
    if x_range > 8
        xtick_interval = 2.0
    elseif x_range > 4
        xtick_interval = 1.0
    elseif x_range > 2
        xtick_interval = 0.5
    else
        xtick_interval = 0.2
    end

    # Generate tick positions - start from 0 if min is close to 0
    if x_min_scaled < xtick_interval
        xtick_start = 0.0
    else
        xtick_start = ceil(x_min_scaled / xtick_interval) * xtick_interval
    end
    xtick_end = floor(x_max_scaled / xtick_interval) * xtick_interval

    # Create tick positions and ensure they're nicely formatted
    xticks_pos = collect(xtick_start:xtick_interval:xtick_end)

    # Format tick labels to remove unnecessary decimals
    xtick_labels = String[]
    for tick in xticks_pos
        if tick == floor(tick)
            push!(xtick_labels, string(Int(tick)))
        else
            push!(xtick_labels, string(tick))
        end
    end

    # First create the scatter plot with a square aspect ratio
    p = scatter(
        x_coords_scaled,
        df_copy.Threshold;
        markersize = marker_sizes,
        marker_z = norm_frequencies,      # Normal frequencies
        color = :cividis,                 # Cividis: blue=low prob, yellow=high prob
        colorbar_title = colorbar_label,
        colorbar_titlefontsize = 10,     # Larger font for colorbar title
        clims = (0, 1),                   # Ensure consistent color scale
        markerstrokecolor = :white,
        markerstrokewidth = 0.3,
        alpha = 0.9,
        label = "",
        xlabel = x_label_text,
        ylabel = "State Uncertainty",
        title = title_text,
        xticks = (xticks_pos, xtick_labels),  # Custom tick positions and labels
        yticks = (0:0.1:1, 0:0.1:1),  # Y-axis from 0 to 1 with 0.1 intervals
        ylims = (-0.05, 1.05),          # Set y-axis limits explicitly with small padding
        titlefontsize = 16,              # Title should be largest
        guidefontsize = 13,              # Axis labels smaller than title
        tickfontsize = 11,               # Tick labels smaller than axis labels
        legendfontsize = 11,             # Legend same as ticks
        grid = (:y, :gray, :dash, 0.3),  # Grid only on y-axis
        aspect_ratio = :auto,            # Let Plots determine aspect ratio
        size = (700, 700),               # Square figure size
        right_margin = 20Plots.mm,       # Standard margin for colorbar only
        left_margin = 5Plots.mm,         # Reduced left margin
        bottom_margin = 7Plots.mm,       # Extra margin for x-axis label with scale
        dpi = 300,
        colorbar = :right,
    )               # Explicitly place colorbar on right

    # Add subtle horizontal bands for visual grouping (optional, very subtle)
    # These help visually separate different uncertainty levels
    x_min = minimum(x_coords_scaled) * 0.98
    x_max = maximum(x_coords_scaled) * 1.02

    for threshold in sort(unique(df_copy.Threshold))
        n_actions = action_counts[threshold]

        # Only add bands if there are multiple action sets
        if n_actions > 2
            # Very subtle band - barely visible
            band_alpha = 0.02 + 0.01 * log1p(n_actions)
            band_alpha = min(0.08, band_alpha)  # Very subtle cap

            # Draw a very thin horizontal band
            plot!(
                p,
                [x_min, x_max],
                [threshold, threshold];
                fillrange = threshold - 0.015,
                fillalpha = band_alpha,
                fillcolor = :gray,
                line = nothing,
                label = "",
            )
            plot!(
                p,
                [x_min, x_max],
                [threshold, threshold];
                fillrange = threshold + 0.015,
                fillalpha = band_alpha,
                fillcolor = :gray,
                line = nothing,
                label = "",
            )
        end
    end

    # Find MLASP points (Maximum Likelihood Action Set Path)
    # We need to carefully select the highest frequency point at each threshold
    # and use the exact same coordinates as the scatter plot
    mlasp_x_coords = Float64[]
    mlasp_y_coords = Float64[]
    mlasp_points = DataFrame()  # Still need this for annotations later

    # Process each unique threshold level
    for threshold in sort(unique(df_copy.Threshold); rev = true)
        threshold_data = filter(row -> row.Threshold == threshold, df_copy)
        if nrow(threshold_data) > 0
            # Find the maximum frequency at this threshold
            max_freq = maximum(threshold_data.Frequency)

            # Get all points with this max frequency (there might be ties)
            max_freq_points = filter(row -> row.Frequency == max_freq, threshold_data)

            # If there are ties, select the first one (consistent selection)
            # This ensures we always pick the same point when there are ties
            best_row = max_freq_points[1, :]

            # Add to mlasp_points for annotations
            push!(mlasp_points, best_row; promote = true)

            # Find this exact point in the original coordinate arrays
            # by matching threshold, frequency, AND the utility value
            found = false
            for i = 1:length(df_copy.Threshold)
                if df_copy.Threshold[i] == best_row.Threshold &&
                   df_copy.Frequency[i] == best_row.Frequency &&
                   df_copy.Average_Utility[i] == best_row.Average_Utility
                    # Use the exact scaled coordinate from the scatter plot
                    push!(mlasp_x_coords, x_coords_scaled[i])
                    push!(mlasp_y_coords, df_copy.Threshold[i])
                    found = true
                    break
                end
            end

            # Fallback if not found (shouldn't happen)
            if !found
                push!(mlasp_x_coords, best_row.Average_Utility / scale_factor)
                push!(mlasp_y_coords, best_row.Threshold)
            end
        end
    end

    # Plot MLASP path as red dashed line connecting the centers of highest frequency points
    # Ensure the line connects the exact points by plotting them in the correct order
    if length(mlasp_x_coords) > 0
        # Sort by y-coordinate (threshold) in descending order to ensure proper line connection
        sorted_indices = sortperm(mlasp_y_coords; rev = true)
        mlasp_x_sorted = mlasp_x_coords[sorted_indices]
        mlasp_y_sorted = mlasp_y_coords[sorted_indices]

        plot!(
            p,
            mlasp_x_sorted,
            mlasp_y_sorted;
            line = (:dashdot, :red, 2.5),  # Dash-dot line for MLASP path
            marker = false,  # No markers - let the line go through scatter points
            label = "MLASP",  # Simplified label
            legend = :topright,  # Move to top-right corner
            foreground_color_legend = nothing,  # Remove legend box
            background_color_legend = nothing,
        )   # Transparent legend background
    end

    # N_i annotations have been removed per user request
    # The plot now only shows the scatter points, MLASP line, and colorbar

    # Add annotations for action set changes (if requested and Action_Set column exists)
    if show_annotations && "Action_Set" in names(df_copy)
        prev_action_set = nothing
        annotations_to_place = NamedTuple[]

        for row in eachrow(mlasp_points)
            current_action_set = row.Action_Set

            # Only annotate if action set changes
            if current_action_set != prev_action_set
                # Clean up the label
                label_text = string(current_action_set)
                label_text = replace(label_text, r"[\[\]'\"\\]" => "")
                if length(label_text) > 30
                    label_text = label_text[1:27] * "..."
                end

                push!(
                    annotations_to_place,
                    (
                        x = row.Average_Utility / scale_factor,  # Scale x-coordinate
                        y = row.Threshold,
                        label = label_text,
                    ),
                )

                prev_action_set = current_action_set
            end
        end

        # Place annotations with smart positioning - right side default, left for edge cases
        if !isempty(annotations_to_place)
            x_range = maximum(x_coords_scaled) - minimum(x_coords_scaled)
            x_min = minimum(x_coords_scaled)
            x_max = maximum(x_coords_scaled)

            # Calculate minimum spacing based on number of annotations
            n_annotations = length(annotations_to_place)
            y_range = maximum(df_copy.Threshold) - minimum(df_copy.Threshold)
            y_min_plot = minimum(df_copy.Threshold)
            y_max_plot = maximum(df_copy.Threshold)
            # Increase minimum spacing to prevent overlaps
            min_spacing = max(0.08, y_range / (n_annotations * 2.0))  # Reduced from 0.12

            # Sort annotations by y position
            sort!(annotations_to_place; by = ann -> ann.y, rev = true)

            # Use a more robust algorithm for positioning
            adjusted_annotations = []

            for (i, ann) in enumerate(annotations_to_place)
                target_y = ann.y

                # Smart positioning: determine if point is near edge
                # Points are normalized, so check relative position
                rel_x_pos = (ann.x - x_min) / x_range  # 0 = left edge, 1 = right edge
                rel_y_pos = (ann.y - y_min_plot) / y_range  # 0 = bottom, 1 = top

                # DEFAULT: Always place annotations horizontally on the right
                # Only move for extreme edge cases to avoid overlaps
                x_offset = x_range * 0.12  # Default 12% offset to the right
                alignment = :left  # Left-aligned text when placed on right

                # Only reposition for true edge cases
                if rel_x_pos > 0.85  # Point is very close to right edge/colorbar (>85%)
                    # This is an edge case - place on left to avoid colorbar
                    x_offset = -x_range * 0.15  # 15% offset to the left
                    alignment = :right
                elseif rel_x_pos > 0.75 && rel_y_pos < 0.1  # Near bottom-right corner
                    # Edge case: bottom-right corner, move left
                    x_offset = -x_range * 0.12
                    alignment = :right
                elseif rel_x_pos > 0.75 && rel_y_pos > 0.9  # Near top-right corner
                    # Edge case: top-right corner, move left
                    x_offset = -x_range * 0.12
                    alignment = :right
                end
                # All other cases keep the default horizontal right placement

                ann_x = ann.x + x_offset

                # Ensure annotation stays within plot bounds
                safe_x_min = x_min + x_range * 0.02  # 2% from left edge
                safe_x_max = x_max - x_range * 0.02  # 2% from right edge
                ann_x = clamp(ann_x, safe_x_min, safe_x_max)

                # Check for conflicts with all previously placed annotations
                conflicts = true
                attempts = 0
                max_attempts = 10

                while conflicts && attempts < max_attempts
                    conflicts = false
                    for prev_ann in adjusted_annotations
                        y_dist = abs(target_y - prev_ann.y)
                        x_dist = abs(ann_x - prev_ann.x)

                        # Check if too close in both x and y directions
                        if y_dist < min_spacing && x_dist < x_range * 0.1
                            conflicts = true
                            # Move up or down based on original position
                            if ann.y > prev_ann.orig_y
                                target_y = prev_ann.y + min_spacing
                            else
                                target_y = prev_ann.y - min_spacing
                            end
                            break
                        end
                    end
                    attempts += 1
                end

                # Ensure we stay within reasonable bounds for y
                y_min = minimum(df_copy.Threshold) - 0.05  # Reduced from 0.1
                y_max = maximum(df_copy.Threshold) + 0.05  # Reduced from 0.1
                target_y = clamp(target_y, y_min, y_max)

                # Store adjusted position with alignment
                push!(
                    adjusted_annotations,
                    (
                        x = ann_x,
                        y = target_y,
                        orig_x = ann.x,
                        orig_y = ann.y,
                        label = ann.label,
                        alignment = alignment,
                    ),
                )
            end

            # Draw annotations with connecting lines
            for ann in adjusted_annotations
                # Draw connecting line from original point to annotation
                plot!(
                    p,
                    [ann.orig_x, ann.x],
                    [ann.orig_y, ann.y];
                    line = (:solid, :gray, 0.3),
                    alpha = 0.2,
                    label = "",
                )

                # Add annotation text with appropriate alignment based on position
                annotate!(p, ann.x, ann.y, text(ann.label, 7, ann.alignment, :darkblue))  # Dynamic alignment
            end
        end
    end

    return p
end
