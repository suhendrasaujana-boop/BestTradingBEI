    # ========== VOLUME PROFILE (FITUR BARU) ==========
    try:
        from volume_profile import calculate_volume_profile
        vp = calculate_volume_profile(df, bins=40, lookback=60)
        if vp and not vp['profile'].empty:
            # Normalisasi volume agar muat di chart
            profile = vp['profile']
            max_vol = profile['volume'].max()
            if max_vol > 0:
                profile['volume_norm'] = profile['volume'] / max_vol * 3  # lebar max 3 candle
                
                # Gambar histogram horizontal
                for _, row in profile.iterrows():
                    if row['volume'] > 0:
                        fig.add_shape(
                            type='rect',
                            x0=len(df_plot) - 1,
                            x1=len(df_plot) - 1 + row['volume_norm'],
                            y0=row['price'],
                            y1=row['price'] + (df_plot['high'].max() - df_plot['low'].min()) / 40,
                            fillcolor='rgba(0, 150, 255, 0.4)',
                            line=dict(width=0),
                            layer='below'
                        )
                
                # Tandai POC dengan garis horizontal
                poc = vp['poc']
                fig.add_hline(
                    y=poc,
                    line_dash='dash',
                    line_color='cyan',
                    opacity=0.8,
                    annotation_text=f'POC: {poc:.0f}',
                    annotation_position='right'
                )
    except Exception as e:
        pass  # Jika gagal, chart tetap muncul tanpa Volume Profile
    # ========== END VOLUME PROFILE ==========
